def translate_sim_data_to_llm_context(time_sim, sim_data_np) -> str:
    """
    AFSIM(C++)发送结构为：simTime 640 (80个平台 * 8个值)
    每个平台 8 个值的含义：
      [0]=live, [1]=lat, [2]=lon, [3]=alt(m), [4]=velN, [5]=velE, [6]=velD, [7]=heading(deg)

    这里只将前 4 个平台（0:red_1, 1:red_2, 2:blue_1, 3:blue_2）整理成 LLM 可读文本。
    """
    try:
        if sim_data_np is None or len(sim_data_np) < 8 * 4:
            return (
                f"T={time_sim:.2f} (state too short: "
                f"{len(sim_data_np) if sim_data_np is not None else 'None'})"
            )

        def plat(idx: int):
            b = 8 * idx
            return {
                "live": int(sim_data_np[b + 0]),
                "lat": float(sim_data_np[b + 1]),
                "lon": float(sim_data_np[b + 2]),
                "alt_m": float(sim_data_np[b + 3]),
                "velN_mps": float(sim_data_np[b + 4]),
                "velE_mps": float(sim_data_np[b + 5]),
                "velD_mps": float(sim_data_np[b + 6]),
                "heading_deg": float(sim_data_np[b + 7]),
            }

        r1 = plat(0)
        r2 = plat(1)
        b1 = plat(2)
        b2 = plat(3)

        return (
            f"T={time_sim:.2f}\n"
            f"red_1: {r1}\n"
            f"red_2: {r2}\n"
            f"blue_1: {b1}\n"
            f"blue_2: {b2}\n"
        )
    except Exception as e:
        return f"Data Error: {str(e)}"

