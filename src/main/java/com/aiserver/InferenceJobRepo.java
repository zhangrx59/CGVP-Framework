package com.aiserver;

import org.springframework.data.jpa.repository.JpaRepository;
import java.util.Optional;

public interface InferenceJobRepo extends JpaRepository<InferenceJob, Long> {
}
