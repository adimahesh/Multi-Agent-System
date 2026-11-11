import time
import streamlit as st


def safe_invoke(runnable, input_data, stage="AI Processing"):
    """
    Run a LangChain runnable with one retry for temporary Gemini server errors.
    Shows a friendly Streamlit message instead of exposing a traceback.
    """
    max_attempts = 2

    for attempt in range(max_attempts):
        try:
            return runnable.invoke(input_data)

        except Exception as exc:
            error_text = str(exc).lower()

            # Authentication errors: a retry cannot repair a rejected key.
            if any(term in error_text for term in (
                "api_key_invalid",
                "invalid api key",
                "api key not valid",
                "401",
                "unauthenticated",
                "access_token_type_unsupported",
            )):
                message = (
                    "🔑 **Google Gemini authentication failed.**\n\n"
                    "Google rejected the API key. Check that your current key is "
                    "complete, active, and enabled for the Gemini API. "
                    "Do not share your key."
                )
                retryable = False

            # Quota/rate limits: do not repeatedly send requests.
            elif any(term in error_text for term in (
                "429",
                "resource_exhausted",
                "rate limit",
                "quota exceeded",
            )):
                message = (
                    "⏳ **Gemini request limit reached.**\n\n"
                    "The API has temporarily limited requests or the project "
                    "has reached its quota. Please wait and check AI Studio usage."
                )
                retryable = False

            # Temporary model/server load.
            elif any(term in error_text for term in (
                "503",
                "unavailable",
                "high demand",
                "overloaded",
                "500",
                "502",
                "504",
                "internal server error",
                "service unavailable",
            )):
                message = (
                    "🚦 **Gemini is experiencing heavy traffic.**\n\n"
                    "The model is temporarily busy. Please try again in a few moments."
                )
                retryable = True

            else:
                message = (
                    f"❌ **{stage} failed.**\n\n"
                    "An unexpected error occurred. Open the technical details "
                    "below if you need to debug it."
                )
                retryable = False

            if retryable and attempt < max_attempts - 1:
                notice = st.empty()
                notice.warning(
                    f"⚠️ {stage}: Gemini is experiencing heavy traffic. "
                    "Retrying once in 5 seconds..."
                )
                time.sleep(5)
                notice.empty()
                continue

            # Ensure the UI does not remain in the running state after failure.
            if "running" in st.session_state:
                st.session_state.running = False

            st.error(message)
            with st.expander("Technical details (for debugging)"):
                st.code(str(exc))

            st.stop()
