# AI assistance record for the Lab 2 appendix

Tool: OpenAI Codex. This log will be updated with concrete actions and the student's contributions as the lab proceeds.

## User prompts in this Lab 2 session

1. "lab2来了 你给我英文介绍一下lab2做啥 一步一步 然后帮我做完"
   English meaning: Lab 2 has been released; explain what to do in English, step by step, and help complete it.
2. "I have clipped in the microSD card into my computer"
3. Hardware confirmation: the card was returned to the Pi with power off.
4. The student chose "60 and 42 — halve both layers" for model C from two explained options.
5. The student requested Chinese explanations, returned the card to the Pi, and reported power-on.
6. Cooling confirmation: "盖子合上，风扇在转" (case closed, fan running).
7. "我们现在这个a b c都是符合lab2要求的对吧" (Are the current A, B and C models all compliant with Lab 2?)
8. Student interpretation: "我觉得参数量最少的最好 因为这些准确率都差不多 然后c参数很少 但是b更快 这和架构有关系" (Prefers the fewest parameters, views these accuracy differences as small, and attributes B being faster than C to architecture.)
9. Student follow-up: "c没有裁剪卷积计算 只是缩小最后的那一层全连接层" (C leaves convolutional computation unchanged and shrinks the final fully connected part.) Codex clarified that both hidden widths change, from 120/84 to 60/42; the output remains ten classes.

## Assistance so far

- Read the complete Tutorial and Assignment in Canvas and the syllabus AI policy.
- Explained the experiment sequence in English.
- Performed a read-only Windows disk check; no microSD formatting or reinstallation.
- Checked and started the existing OMEN hotspot without changing its name or password.
- Cloned and inspected the instructor's Lab 2 repository locally; no training or reported inference has run on OMEN.
- Created a requirements checklist. No final results or report conclusions have been written.
- Transcribed `LeNet.forward` and `train_step` directly from Tutorial Steps 2 and 3 into a working copy; these function bodies are supplied course code, not a newly designed AI solution.
- Filled B's dictionary using the assignment's required widths and C's dictionary using the student's chosen 60/42 widths. All reference training/evaluation/benchmark code remains unchanged.
- Connected by SSH to the actual supplied Pi; checked its model, OS/Python, storage, temperature, power flags, and network.
- Started installation of the instructor's pinned requirements in a separate Pi virtual environment.
- Prepared an external orchestration helper to run the unmodified instructor commands, preserve console logs, record stage times and telemetry, and pause for cooling between training runs and inference benchmarks. This helper is AI-written operational support, not part of the supplied model or timing implementation.
- Confirmed all three model checks passed on the Pi with parameter counts 44,426, 27,180 and 20,984. Fixed and recorded the designs before training or test evaluation.
- Recovered a dropped wireless connection by restarting the existing OMEN hotspot; its name, password and 5 GHz band were unchanged. Started the experiment detached from SSH so a transient wireless disconnect cannot terminate the experiment.
- Prepared a local evidence audit that checks transferred file hashes, unchanged instructor source, the recorded Pi/CPU/thread settings and statistics against original CSV data. This operational helper performs no model inference or training.
- Completed every required training/evaluation/benchmark on the Pi. The instructor summarizer passed; exported source/evidence passed 144 audit checks on OMEN. The three original timing CSVs each retain all 100 observations.
- Translated and edited the student's C preference and architecture explanation into the report, quantified accuracy differences in percentage points, and added the required training-time comparison and measurement conditions from verified data.
- Created a two-page XeLaTeX PDF: one page of analysis and one AI Attribution Appendix. No Canvas submission has been made.
- Visually inspected both rendered pages and checked the page count and numerical table. Copied the final PDF/LaTeX back to the Pi, verified original evidence hashes, downloaded and verified the complete archive on OMEN, and successfully invoked operating-system shutdown. The student confirmed "绿灯已停止闪烁" (green light stopped flashing).

## Student contributions to record

- Physical card installation and Pi power/cooling setup.
- Model C architecture choice: 60/42, halving both baseline fully connected widths while keeping convolution unchanged.
- Interpretation after seeing measured results: choose C because parameter count is the priority and the accuracy differences are acceptable; C preserves convolution while shrinking the fully connected part, so architecture explains the runtime tradeoff. The report makes the 0.88 percentage-point loss vs. A explicit and does not claim statistical equivalence.

The final report must accurately disclose any additional coding, experiment execution, analysis, editing or LaTeX assistance. Do not claim the student authored unadapted AI-generated text or code.
