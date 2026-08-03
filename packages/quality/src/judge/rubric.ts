import type { TaskType } from "@orchestrator/shared";

/**
 * Per-task grading rubrics.
 *
 * Each asks for a single number and nothing else. Judges that are invited to explain themselves
 * produce longer, costlier, more variable output, and the explanation is not used — the reward
 * function consumes one scalar.
 */
const RUBRICS: Record<TaskType, string> = {
  general:
    "Does the response answer what was asked, accurately and completely, without padding or evasion?",
  code: "Is the code correct, complete, and idiomatic? Would it run as written for the stated task?",
  extraction:
    "Was every requested field extracted, accurately, with nothing invented that is absent from the source?",
  summarization:
    "Does the summary preserve the key points faithfully, at appropriate length, without adding claims?",
  classification:
    "Is the label correct and drawn from the permitted set, with no hedging or extra commentary?",
  reasoning:
    "Is the reasoning valid and the conclusion sound? Check the steps, not merely the final answer.",
  creative:
    "Does it satisfy the brief with genuine craft — specific, coherent, and not generic filler?",
};

export const JUDGE_SYSTEM_PROMPT = [
  "You grade AI responses for a routing system. You will see a task, a user request, and a response.",
  "",
  "Reply with ONLY a number from 0.0 to 1.0 — no words, no punctuation, no explanation.",
  "",
  "  0.0  unusable: wrong, empty, or ignores the request",
  "  0.3  seriously deficient but on topic",
  "  0.5  partially correct, notable gaps",
  "  0.7  correct with minor issues",
  "  0.9  fully correct and well executed",
  "  1.0  exemplary",
  "",
  "Grade only what is in front of you. Do not reward length, confidence, or formatting.",
].join("\n");

export function buildJudgePrompt(
  taskType: TaskType,
  requestText: string,
  responseText: string,
): string {
  return [
    `Task type: ${taskType}`,
    `Criterion: ${RUBRICS[taskType]}`,
    "",
    "--- REQUEST ---",
    truncate(requestText, 4_000),
    "",
    "--- RESPONSE ---",
    truncate(responseText, 4_000),
    "",
    "Score (a number between 0.0 and 1.0, nothing else):",
  ].join("\n");
}

/**
 * Parse the judge's reply into a score.
 *
 * Returns undefined rather than guessing when the reply is not a usable number. A judge that ignored
 * its instructions has told us nothing, and defaulting to some middle value would quietly inject
 * noise into the signal this whole phase exists to make trustworthy.
 */
export function parseJudgeScore(reply: string): number | undefined {
  const match = reply.trim().match(/(\d*\.?\d+)/);
  if (!match?.[1]) return undefined;

  const value = Number.parseFloat(match[1]);
  if (!Number.isFinite(value)) return undefined;
  // Tolerate a judge that answers on a 0-100 scale despite the rubric.
  const normalized = value > 1 && value <= 100 ? value / 100 : value;
  if (normalized < 0 || normalized > 1) return undefined;

  return normalized;
}

function truncate(text: string, limit: number): string {
  return text.length <= limit ? text : `${text.slice(0, limit)}\n…[truncated]`;
}
