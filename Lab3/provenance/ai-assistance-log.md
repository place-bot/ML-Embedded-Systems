# Lab 3 assistance record

Tool: OpenAI Codex. User preference: Chinese, brief confirmed hardware steps, LaTeX report, concise accurate AI attribution, retained code/results in the existing course repository.

Initial prompt: "lab3出了 microsd放进去了 okta也登录了" (Lab 3 is available; the microSD has been inserted and Okta is logged in).

Actions completed on 2026-09-29:

- Read the entire Lab 3 Assignment (2 pages), Tutorial (4 pages), and syllabus AI policy in authenticated Canvas.
- Recorded the discrepancy between the Canvas deadline (October 4, 23:59) and PDF wording (October 1 before class).
- Performed read-only disk inspection: OMEN currently showed the 32 GB USB card and D: bootfs; no card contents were changed.
- Checked hotspot state: off, 2.4 GHz, no clients at the initial check.
- Asked the student to eject the card from OMEN and return it to the unpowered Pi before further hardware steps.
- Cloned and inspected instructor source commit 4c635083c39862381940db06d134bfe95ce92088. No implementation exercises are required; all instructor code is to remain unchanged.
- Verified the retained Lab 2 A checkpoint hash on OMEN against its training metadata. No model execution or reported timing has run on OMEN.

- The student confirmed the card was returned to the Pi and power connected, with the case closed and fan running. Later reported red LED on and green LED off. This alone does not establish successful boot.
- Started the existing OMEN hotspot without changing its SSID or password. Tested fixed 5 GHz, then 2.4 GHz; at the checks so far there were no clients and SSH at the previous Pi address timed out. No Lab 3 model execution has started.

- Following one user-performed power cycle, the green LED flashed and the Pi reconnected on 2.4 GHz at 192.168.137.101. SSH succeeded with the existing known host key.
- Read-only Pi checks at 2026-09-29 17:46 UTC verified Raspberry Pi 4B Rev 1.5, aarch64, Debian 13.5, Python 3.13.5, 19 GB free, 37 C, throttling flags 0x0, and the unchanged own Lab 2 A checkpoint hash. No existing Lab 3 directory was present.
- Prepared external setup/execution helpers that call the unmodified instructor commands sequentially and record provenance. They are orchestration, not changes to instructor model/pruning/timing implementations.

- The independent Lab 3 virtual environment, pinned dependencies, CPU forward/backward check, and official 12000/2000/10000 dataset preparation succeeded on the Pi.
- Import reported PASS source with validation 79.65% and fallback=False. Both supplied pruning checks passed. Validation immediately after pruning was Control 79.65%, S25 77.45%, S50 64.05%.
- All three fine-tuning runs completed three epochs, followed by full test evaluations and PASS reload for all three. Final validation: 82.60%, 82.35%, 81.40%; test accuracy: 81.77%, 81.70%, 81.00%, in Control/S25/S50 order.
- The detached pipeline continued through SSH interruptions. Control timing reported median 2.6906 ms (100 records). The last successful status observation showed S25 benchmarking in progress. Remaining timing and overall completion have NOT yet been verified.
- SSH/hotspot disconnected again. Fixed 5 GHz did not reconnect; hotspot was returned to 2.4 GHz and left on. Requested physical confirmation that the Pi still has a red light and running fan. No second power cycle has been requested or performed by the agent.
- Evidence collection and transfer requests have not succeeded yet; there is no confirmed Lab 3 result archive on OMEN. External evidence-audit and LaTeX-builder scripts are prepared. The PDF authoring operation marker has been run once, but no report PDF has been generated; student deployment choice is still needed after complete timing results are available.

- After the student independently restarted the Pi, SSH recovered. The existing detached run had completed at 2026-09-29 17:55:04 UTC, before that restart. No training or timing was repeated.
- Collected the original evidence archive to OMEN and verified SHA-256 dcc17b3091f35dbeb3181f9a7bd24fd1d7254ba94dee3d9da1d0435392336b0b. Read-only local evidence verification passed 186 checks.
- Final inference medians (ms): Control 2.6906365, S25 2.6027935, S50 2.507561. Parameter counts are 44,426 / 36,142 / 27,858. All three timing CSVs contain exactly 100 positive records. Pre-warmup temperatures span 41.381-43.329 C, with frequency 1.8 GHz and throttled=0x0 at each sampled benchmark boundary.
- Diagnosed Wi-Fi power saving as enabled. After all experiments were complete, disabled it in the active NetworkManager profile and live wlan0 interface as a reversible connectivity troubleshooting measure; root cause of disconnections remains unproven. Network-setting changes do not alter the retained experiment evidence.
- Asked the student for their deployment model preference and interpretation of the mismatch between parameter and latency reductions. This response is pending.

- Executed sudo poweroff successfully at 2026-09-29 18:09:22 UTC (14:09 local); command returned exit code 0 and SSH disconnected. Closed the OMEN SSH helper. Student confirmation that green activity has stopped is pending.
- The experiment and evidence backup are complete. The report has not yet been generated because the student's personal analysis is still pending. It can be completed entirely on OMEN; no new Pi run or reboot is required.

- The student confirmed that the green activity LED stopped. Safe shutdown is complete; the Pi does not need to restart for report work.
- The student selected S50 because it has the fewest parameters. They asked whether inference mainly meant the final output layer. Clarified that inference includes the entire network and that aggregate timings cannot identify a final-layer bottleneck. The report preserves the student's size priority and presents the corrected technical interpretation with AI assistance disclosed.

- Generated the two-page XeLaTeX report (one page of analysis and one concise AI Attribution Appendix). Inspected both rendered pages and verified the extracted text against the recorded comparison results. The report includes the student's S50 choice and a corrected whole-network explanation of inference cost, without claiming an unmeasured output-layer bottleneck.

Report, experiment, original-evidence backup and safe shutdown are complete. GitHub publication is handled as a separate repository commit; the retained raw data are unchanged.
