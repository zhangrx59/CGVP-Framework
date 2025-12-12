# -*- coding: utf-8 -*-
"""
异步协程版：Qwen-VL 批量生成皮损性状描述（支持断点续跑）

- 使用 asyncio + AsyncOpenAI 实现真正的并发调用
- 并发数量由 config["analysis_config"]["max_workers"] 控制（默认最多 16）
- 自动跳过已完成行（lesion_shape 非空且非 error）
- 只处理当前图片文件夹中能找到图片的样本
"""

import os
import base64
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple

import asyncio
import pandas as pd
from openai import AsyncOpenAI


# ===================== 通用工具 =====================

def load_config(config_path: str = "config.json") -> Dict[str, Any]:
    """从 JSON 文件加载配置"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def encode_image_to_base64(image_path: Path) -> str:
    """将本地图像转换为 base64 编码"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def build_imageid_to_path(folder_path: str, supported_formats) -> Dict[str, Path]:
    """
    扫描图片文件夹，构建 {image_id(不带扩展名) -> 完整路径} 映射
    只要文件名是 XXX.jpg / XXX.png，就对应 image_id = XXX
    """
    folder = Path(folder_path)
    mapping: Dict[str, Path] = {}

    if not folder.exists():
        raise FileNotFoundError(f"图片文件夹不存在: {folder}")

    for f in folder.iterdir():
        if f.is_file() and f.suffix.lower() in supported_formats:
            image_id = f.stem  # 去掉扩展名
            mapping[image_id] = f

    print(f"📁 在 {folder_path} 中共找到 {len(mapping)} 个 image_id 对应的图片文件")
    return mapping


def safe_read_table(path: str) -> pd.DataFrame:
    """
    既支持 csv 也支持 xlsx：
    - .csv 用多种编码尝试
    - .xls/.xlsx 用 read_excel
    """
    ext = Path(path).suffix.lower()
    if ext in [".xls", ".xlsx"]:
        print(f"🧾 检测到 Excel 文件：{path}，使用 read_excel 读取")
        return pd.read_excel(path)

    encodings_to_try = ["utf-8", "gbk", "gb2312", "latin1"]
    last_err = None
    for enc in encodings_to_try:
        try:
            print(f"尝试使用编码 {enc} 读取 CSV...")
            df = pd.read_csv(path, encoding=enc, low_memory=False)
            print(f"✅ 使用编码 {enc} 读取 CSV 成功")
            return df
        except UnicodeDecodeError as e:
            print(f"❌ 使用编码 {enc} 失败: {e}")
            last_err = e

    raise last_err


# ===================== 异步调用 Qwen =====================

async def call_qwen_shape_async(
    client: AsyncOpenAI,
    model_type: str,
    image_path: Path,
    shape_prompt: str,
    max_retries: int = 3,
    base_retry_delay: float = 15.0,
) -> str:
    """
    异步调用 Qwen-VL，返回“病变性状描述”的简短中文。

    - 带有限流友好的重试逻辑（429 / TPM limit）
    - 不做随机 sleep，而是指数回退等待
    """
    base64_image = encode_image_to_base64(image_path)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": shape_prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    },
                },
            ],
        }
    ]

    for attempt in range(1, max_retries + 1):
        try:
            resp = await client.chat.completions.create(
                model=model_type,
                messages=messages,
                stream=False,
            )

            content = resp.choices[0].message.content

            if isinstance(content, list):
                texts = []
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        texts.append(part.get("text", ""))
                    elif isinstance(part, str):
                        texts.append(part)
                result = "".join(texts).strip()
            else:
                result = str(content).strip()

            # 只保留第一行，去掉多余空白
            result = result.splitlines()[0].strip()
            return result

        except Exception as e:
            err_str = str(e)
            # 简单判断是否是限流 / 429 / TPM
            if "429" in err_str or "rate limit" in err_str or "TPM limit" in err_str:
                if attempt < max_retries:
                    delay = base_retry_delay * attempt
                    print(f"⚠️ 检测到限流，{attempt}/{max_retries} 次，等待 {delay:.1f}s 后重试...")
                    await asyncio.sleep(delay)
                    continue
            # 非限流错误 或 最后一轮重试失败，直接抛出
            raise


# ===================== 异步 worker =====================

async def worker_task_async(
    task_idx: int,
    row_idx: int,
    image_id: str,
    image_path: Path,
    client: AsyncOpenAI,
    model_type: str,
    shape_prompt: str,
    semaphore: asyncio.Semaphore,
) -> Tuple[int, str, str, str]:
    """
    单个异步任务：
    - 使用信号量控制同时在飞的请求数（并发数）
    - 调用 call_qwen_shape_async
    - 返回 (row_idx, image_id, shape, error)
    """
    async with semaphore:
        try:
            print(f"🧵[任务{task_idx}] 开始处理 image_id={image_id}, 文件={image_path.name}")
            shape = await call_qwen_shape_async(client, model_type, image_path, shape_prompt)
            return row_idx, image_id, shape, ""
        except Exception as e:
            err_str = str(e)
            print(f"❌[任务{task_idx}] image_id={image_id} 调用失败: {err_str}")
            return row_idx, image_id, f"error:{err_str}", err_str


# ===================== 主流程（异步） =====================

async def annotate_image_shapes_async(
    config_path: str = "config.json",
    output_path: str = None,
    save_every_n: int = 20,
):
    """
    主逻辑：
    - 读取表格 & 图片列表
    - 检测哪些行已经完成 / 未完成
    - 对未完成部分构建任务列表
    - 使用 asyncio 并发调 Qwen，填充 lesion_shape
    """
    config = load_config(config_path)
    api_config = config["api_config"]
    analysis_config = config["analysis_config"]
    prompts_config = config.get("prompts", {})

    folder_path = analysis_config["folder_path"]
    table_path = analysis_config["csv_file_path"]
    supported_formats = [s.lower() for s in analysis_config["supported_formats"]]

    image_id_column = analysis_config.get("image_id_column", None)
    shape_prompt = prompts_config.get(
        "shape_prompt",
        "你是一名皮肤科医生，请用中文简要描述皮损性状。"
    )

    # 1. 读取表格（CSV / Excel）
    df = safe_read_table(table_path)
    print(f"📊 成功加载表格，记录数：{len(df)}")
    print(f"当前列名：{df.columns.tolist()}")

    # 自动检测 image_id 列
    if image_id_column is None or image_id_column not in df.columns:
        if "图片ID" in df.columns:
            image_id_column = "图片ID"
            print("ℹ️ 未在 config 中找到可用的 image_id_column，自动使用列名：图片ID")
        else:
            raise ValueError(
                f"在表格中未找到配置的 image_id_column，也不存在“图片ID”这一列，请检查：现有列为 {df.columns.tolist()}"
            )
    print(f"👉 将使用列 {image_id_column!r} 作为图片 ID 列")

    # lesion_shape 列
    if "lesion_shape" not in df.columns:
        df["lesion_shape"] = ""
        print("🆕 新增列 lesion_shape 用于保存病变性状描述")
    else:
        print("ℹ️ 已存在列 lesion_shape，将在其基础上补全/覆盖空值")

    # 断点续跑：统计完成 / 未完成
    col = df["lesion_shape"].astype(str)
    finished_mask = col.str.strip().ne("") & ~col.str.startswith("error:")
    unfinished_mask = ~finished_mask

    finished_count = int(finished_mask.sum())
    unfinished_count = int(unfinished_mask.sum())
    print(f"✅ 已完成样本（lesion_shape 非空且非 error）: {finished_count}")
    print(f"⏳ 未完成样本（空白或 error，将继续诊断）: {unfinished_count}")

    if unfinished_count == 0:
        print("🎉 所有行都已经有 lesion_shape 结果，无需继续诊断。")
        return

    # 2. 构建 image_id -> image_path 映射
    imageid_to_path = build_imageid_to_path(folder_path, set(supported_formats))

    # 3. 构建任务列表（只处理 未完成 且 有图片 的行）
    tasks_meta: List[Tuple[int, str, Path]] = []
    for row_idx, row in df[unfinished_mask].iterrows():
        image_id = str(row[image_id_column])
        image_path = imageid_to_path.get(image_id)

        if image_path is None:
            print(f"⚠️ row_idx={row_idx}, image_id={image_id} 在文件夹中找不到对应图片，标记为 image_not_found")
            df.at[row_idx, "lesion_shape"] = "image_not_found"
            continue

        tasks_meta.append((row_idx, image_id, image_path))

    total_tasks = len(tasks_meta)
    print(f"🧾 本次实际需要调用大模型的样本数：{total_tasks}")
    if total_tasks == 0:
        print("⚠️ 未完成的行中，没有任何一行能在图片文件夹中找到图片。直接保存退出。")
        out = output_path or table_path
        df.to_csv(out, index=False, encoding="utf-8-sig")
        print(f"📄 结果已保存到：{out}")
        return

    # 4. 创建 AsyncOpenAI 客户端（多 key）
    api_keys: List[str] = api_config["api_keys"]
    if len(api_keys) == 0:
        raise RuntimeError("config.json 中 api_config.api_keys 为空，请配置至少一个 key。")

    max_workers_conf = analysis_config.get("max_workers", 16)
    # 这里的 num_workers 是最大并发请求数（可以根据实际限流情况适当调小，比如 8）
    num_workers = min(16, len(api_keys), max_workers_conf)
    print(f"🔑 将使用 {num_workers} 个 API key，并发上限 = {num_workers}")

    used_keys = api_keys[:num_workers]
    clients: List[AsyncOpenAI] = [
        AsyncOpenAI(api_key=k, base_url=api_config["base_url"]) for k in used_keys
    ]

    semaphore = asyncio.Semaphore(num_workers)

    # 5. 构建所有异步任务
    start_time = time.time()
    coros = []
    for task_idx, (row_idx, image_id, image_path) in enumerate(tasks_meta):
        client = clients[task_idx % num_workers]
        coro = worker_task_async(
            task_idx=task_idx,
            row_idx=row_idx,
            image_id=image_id,
            image_path=image_path,
            client=client,
            model_type=api_config["model_type"],
            shape_prompt=shape_prompt,
            semaphore=semaphore,
        )
        coros.append(coro)

    # 6. 并发执行 + 结果写回 DataFrame
    completed = 0
    for fut in asyncio.as_completed(coros):
        row_idx, image_id, shape, err = await fut
        df.at[row_idx, "lesion_shape"] = shape
        completed += 1

        if err == "":
            print(f"✅ [{completed}/{total_tasks}] image_id={image_id} 形状: {shape}")
        else:
            print(f"❌ [{completed}/{total_tasks}] image_id={image_id} 出错: {err}")

        if completed % save_every_n == 0:
            tmp_out = output_path or table_path.replace(".csv", "_with_shape.csv")
            df.to_csv(tmp_out, index=False, encoding="utf-8-sig")
            print(f"💾 已保存中间结果到 {tmp_out}")

    end_time = time.time()

    # 7. 关闭所有 AsyncOpenAI 客户端的底层会话
    for c in clients:
        await c.close()

    # 8. 最终保存
    final_out = output_path or table_path.replace(".csv", "_with_shape.csv")
    df.to_csv(final_out, index=False, encoding="utf-8-sig")
    print(f"\n🎉 本次未完成部分全部诊断完成！")
    print(f"📄 结果已保存到：{final_out}")
    print(f"⏱️ 本次处理耗时: {end_time - start_time:.2f} 秒，平均每个样本 {(end_time - start_time) / max(total_tasks, 1):.2f} 秒")


# ===================== 同步入口 =====================

if __name__ == "__main__":
    print("🚀 启动皮肤病变“性状”标注程序（异步协程版，断点续跑）...")
    asyncio.run(annotate_image_shapes_async("../config.json"))
    print("✅ 全部完成。")
