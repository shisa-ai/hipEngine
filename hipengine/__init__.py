"""Public hipEngine API.

Importing this module must remain torch-free. Optional torch interop belongs behind the
``hipengine[torch]`` extra and outside the runtime hot path.
"""

from hipengine.execution_profiles import ExecutionProfile
from hipengine.llm import LLM, SamplingParams

__all__ = ["ExecutionProfile", "LLM", "SamplingParams"]
