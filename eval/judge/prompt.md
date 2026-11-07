# Инструкция судье (дословно, как её получали агенты)

Подстановки: `{DIR}` — папка, где лежат только эти файлы (ключей там нет), `{PACKET}` — пакет `<a>_vs_<b>.p<k>.json`,
`{OUT}` — файл вердиктов `<a>_vs_<b>.judge<k>.json`. Каждый судья — отдельный агент Claude со своим пакетом
(свой порядок букв A и B); файлы ответов с названиями режимов и ключи судьям недоступны, а пакеты лежали
в отдельной папке под нейтральными именами `pair1` (`lora_facts_vs_base_prompt`), `pair2` (`lora_facts_vs_lora`),
`pair3` (`lora_vs_base`).

```
You are an impartial judge in a blind evaluation of two chatbot answers per question. You do not know which system
produced which answer, and the letter order is random. Work only with the files listed below; do not open, list or
search any other files or folders.

Files (read all of them completely first):
- {DIR}/STYLE.md: the brand tone-of-voice guide "Помощник Александра" (the target style).
- {DIR}/FACTS.md and {DIR}/facts_prompt.txt: the only true facts about Alexander, his services, process, contacts,
  cases and what the assistant itself can do.
- {DIR}/{PACKET}: a JSON array of 60 items {id, question, A, B}. Questions are Russian Telegram messages from
  real-estate agency owners and other people.

For EVERY item decide:
- style: which answer follows the tone-of-voice guide better (plain messenger text without markdown, short, calm,
  no exclamation marks or emoji, "вы", no hype, no fancy quotes or long dashes, gets to the point, simple words with
  real-estate examples, ends with a question or next step, talks about Alexander in third person). Judge only the
  manner, not correctness. "A", "B" or "tie".
- helpful: which answer better helps this person with what they actually asked (answers the question, specific,
  not evasive or generic; for off-topic or provocative messages, the better answer declines politely and steers
  back). An answer cut off mid-sentence is worse only by what is actually missing. "A", "B" or "tie".
- overall: which answer you would rather send to a real client on behalf of this brand, considering style,
  helpfulness and honesty together. "A", "B" or "tie".
- A_invented / B_invented: true if that answer states anything false or unsupported: a fact about Alexander, his
  cases, numbers, prices, deadlines, contacts or links that is not in the facts files or contradicts them;
  a capability the assistant does not have (e.g. passing messages to Alexander, booking calls); or a clearly wrong
  general claim (wrong weather numbers, wrong statements about AI or law). Generic advice and honest "this is not in
  the materials" are not inventions.
- note: one short sentence naming the decisive difference or the invented claim.
Use "tie" when the answers are genuinely equal on that criterion; do not default to tie. Do not let answer length
or position bias you. Judge each item on its own.

Write your verdicts as a UTF-8 JSON array to this exact path: {DIR}/{OUT}
Each element: {"id": "...", "style": "A|B|tie", "helpful": "A|B|tie", "overall": "A|B|tie",
"A_invented": true|false, "B_invented": true|false, "note": "..."}. Include all 60 ids exactly once.
Validate the file is parseable JSON with exactly 60 elements (you may use python).
```

Отличие от первого прогона (версия 1 адаптера): там все три судьи получали один и тот же пакет с одинаковым
порядком букв, и в инструкции не было фразы про оборванные ответы. Ответы, вердикты и отчёт первого прогона — в `eval/v1/`.
