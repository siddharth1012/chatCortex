"""OpenAI chat-completions execution backend (optional ``real`` extra).

Installation: ``pip install chatcortex[real]`` — this module imports
``openai`` lazily inside :func:`OpenAIBackend.__init__`, so the package
never imports it unless a backend is actually built. The API key comes
from ``OPENAI_API_KEY`` (the SDK's standard env resolution); tests can
inject a stub ``client`` instead and need no key or network.

What is measured vs. declared:

- **Latency** and **success** are measured from the real SDK call.
- **Cost** is not derivable from a completion alone (token pricing is
  model-specific and changes over time), so the component's declared
  ``cost_per_call`` is recorded. Component metadata stays pure: nothing
  here mutates it, calibration is the component that turns telemetry
  into updated metadata.
"""

import time
from typing import Any, Callable, Optional

from ..execution.executor import ProbeResult
from ..registry.metadata import ComponentMetadata

DEFAULT_MODEL = "gpt-5.5-mini"


class OpenAIBackend:
    """Runs one graph node as one chat-completions API call.

    Args:
        model: Model name used for every component; ignored when
            ``model_for`` is supplied.
        model_for: Optional ``Callable[[ComponentMetadata], str]`` that
            maps a component to the model it should run on (e.g. cheap
            models for retrieval tooling, stronger models for
            generation); takes precedence over ``model``.
        client: Optional pre-built OpenAI client (or any object exposing
            ``chat.completions.create``). Meant for tests; when given,
            the ``openai`` package is never imported.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        model_for: Optional[Callable[[ComponentMetadata], str]] = None,
        client: Optional[Any] = None,
    ):
        self._model = model
        self.model_for = model_for
        if client is not None:
            self._client = client
        else:
            try:
                import openai
            except ImportError as exc:  # pragma: no cover - exercised only without extra
                raise ImportError(
                    "OpenAIBackend requires the optional 'openai' dependency; "
                    "install it with: pip install chatcortex[real]"
                ) from exc
            self._client = openai.OpenAI()

    def _model_for_component(self, metadata: ComponentMetadata) -> str:
        if self.model_for is not None:
            return self.model_for(metadata)
        return self._model

    def __call__(
        self,
        metadata: ComponentMetadata,
        node_input: Any = None,
    ) -> ProbeResult:
        model = self._model_for_component(metadata)
        prompt = str(node_input) if node_input is not None else (
            f"Perform the '{metadata.capabilities[0]}' task as component "
            f"'{metadata.name}'."
        )

        started = time.perf_counter()
        try:
            response = self._client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
            )
        except Exception as exc:  # network/HTTP/key errors are probe failures
            latency_ms = (time.perf_counter() - started) * 1000.0
            return ProbeResult(
                success=False,
                latency_ms=latency_ms,
                cost=0.0,
                output=f"openai call failed: {type(exc).__name__}: {exc}",
            )

        latency_ms = (time.perf_counter() - started) * 1000.0
        output = response.choices[0].message.content
        return ProbeResult(
            success=bool(output),
            latency_ms=latency_ms,
            cost=metadata.cost_per_call,
            output=output,
        )
