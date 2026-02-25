# AGENTS.md

## General
- You are datascience coding assistant
- Follow best practices for Python coding, data science, and machine learning
- Prioritize correctness, clarity, and reproducibility over cleverness
- Prefer simple, explicit code over abstractions
- Assume code will be read by researchers, not only engineers
- For new functions add concise informative docstrings
- Use typing in function signatures
- If instructions are not perfectly clear, ask for clarification before proceeding
- Parameters should be configurable (no hardcoded paths or constants); if you want to hardcode something, ask for clarification first
- You can hardcode values only in jupyter notebooks for exploration and visualization, never in scripts or modules
- Do not write duplicate code; reuse when possible; if needed, create helper functions or classes
- Log intermediate results when useful
- Minimize dependencies
- Do not introduce new libraries unless clearly beneficial
- Make randomness, I/O, and assumptions explicit
- Avoid data leakage and hidden side effects
- Prefer readable vectorization over premature optimization
- When plotting confusion matrices, we should normalize each row to show per-class percentages (values between 0.0 and 1.0) and fix the color scale to the [0, 1] range.

## Conda environment
- We always use conda environments.
- When using python, first activate conda environment named `moment4ET`.

## Hardware
- I have NVIDIA GeForce RTX 4070 with 12282MiB memory
- I use Ubuntu
- The important snippet from the output of `sudo lshw -short` is as follows:
Device           Class          Description
===========================================
                 system         B550M-ITX/ac (To Be Filled By O.E.M.)
                 bus            B550M-ITX/ac
                 memory         64KiB BIOS
                 memory         32GiB System Memory
                 memory         16GiB DIMM DDR4 Synchronous Unbuffered (Unregistered) 3600 MHz (0.3 ns)
                 memory         16GiB DIMM DDR4 Synchronous Unbuffered (Unregistered) 3600 MHz (0.3 ns)
                 memory         512KiB L1 cache
                 memory         4MiB L2 cache
                 memory         32MiB L3 cache
                 processor      AMD Ryzen 7 5700X 8-Core Processor
                 bridge         Starship/Matisse Root Complex
                 bridge         Starship/Matisse GPP Bridge
/dev/nvme0       storage        Samsung SSD 990 PRO 2TB
hwmon0           disk           NVMe disk
/dev/ng0n1       disk           NVMe disk
/dev/nvme0n1     disk           2TB NVMe disk
/dev/nvme0n1p1   volume         1074MiB Windows FAT volume
/dev/nvme0n1p2   volume         1861GiB EXT4 volume

## Project Info
- This repo works on eye-tracking data. 
- It implements the MOMENT FM on our eye-tracking data.
- Our pimary goal is to compute embeddings for our eye-tracking data. We will later feed them to other ML models.