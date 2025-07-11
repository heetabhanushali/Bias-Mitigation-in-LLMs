def apply_context_injection(prompts: list[str], strategy: str = "context_injection", debug=False) -> list[str]:
    if not prompts:
        return []

    if strategy == "context_injection":
        mitigated = [f"As an unbiased AI, please respond: {p}" for p in prompts]

    # elif strategy == "bias_alert_suffix":
    #     mitigated = [f"{p} (Please respond without any bias or stereotypes.)" for p in prompts]

    else:
        mitigated = prompts  # fallback to original

    if debug:
        print("\n--- context injection Examples ---")
        for i in range(min(3, len(prompts))):
            print(f"Original : {prompts[i]}")
            print(f"Mitigated: {mitigated[i]}\n")

    return mitigated
