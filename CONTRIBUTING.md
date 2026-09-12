# Contributing to BEEP

Contributions go through pull requests against `main`. Keep changes focused,
add or update tests for behavior you change, and add a `CHANGELOG.md` entry
under `[Unreleased]` for anything user-facing.

## Commit authorship

Use whatever tools help you. The authors of a commit have to be people.

GitHub builds the contributor list from three fields: the commit author, the
committer, and any `Co-authored-by:` trailer. An assistant in any of them
becomes a contributor to BEEP, and a squash merge copies the branch's trailers
into the permanent history, where removing them means rewriting it. The
trailer is the one that catches people out, because a tool can add it while
the author field stays correctly your own.

Saying that a tool helped is fine and is not what this is about. A
`Made-with:` trailer, a generated-with footer, a sentence in the message body:
all allowed. The line is authorship, not disclosure.

The `attribution` check enforces this on the commits your PR adds. To see what
it sees:

```bash
python3 .github/scripts/check_ai_attribution.py origin/main HEAD
```

If it catches something, drop the line and amend (`git commit --amend` for the
tip, `git rebase -i` further back). Claude Code stops adding the trailer with
`includeCoAuthoredBy: false`; other tools have an equivalent setting.

A contributor whose own name is Claude is unaffected. The check keys on vendor
addresses, bot accounts and product names such as `Claude Opus`, never on a
bare first name.
