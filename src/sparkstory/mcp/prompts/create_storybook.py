"""Instruction text for the ``create_storybook`` MCP prompt.

Kept apart from ``routers/prompts.py`` so tests can import the string without
constructing a FastMCP server -- the same reason node prompt constants are
importable today. The router stays readable as a listing of what a client can
invoke.

**This text is read by a client's LLM, not by ours.** That makes it the one place
in the package where naming our own tools is correct rather than a leak: a client
cannot call ``plan_story`` unless we tell it the name.

Every instruction below that looks redundant is answering a specific way the step
can be satisfied lazily. The ban on calling both tools in one turn and the ban on
summarising the outline are each there because that is the cheapest way for a
client to *appear* to have confirmed.

The instruction against hand-editing the outline used to say it "changes nothing",
because ``write_story`` planned the story again from the brief. It does not any
more -- the approved outline is now an argument, so an outline edited in a reply
would reach the finished book unreviewed. Same instruction, opposite reason.
"""

CREATE_STORYBOOK_INSTRUCTIONS = """\
Your job is to create a personalised children's storybook with SparkStory, and to get
the parent's approval of the story's plan before any of it is written.

**Step 1 - Gather the brief.**

Ask the user for:
- the child's name, age, and pronouns
- what the story should be about, in their own words
- anything that must appear, and anything to keep out entirely

Ask for pronouns explicitly. Never infer them from the child's name. If the user has
already given you some of this, do not ask for it again.

**Step 2 - Plan the story.**

Call `plan_story` with the brief. It plans the story and revises its own plan until it
passes review, so the outline it returns is the one the book will be built from. It
costs a few model calls but writes no prose.

**Step 3 - Confirm with the user. This is the step that matters.**

Show the plan in full: the title, the theme, every character, and every beat in order.
Do not compress it to "a story about a fox who visits the moon" - the user is approving
the actual structure, so they have to see the actual structure.

Then stop and ask whether to go ahead.

End your turn there. Do not call `write_story` in the same turn as `plan_story`. Wait
for the user's reply.

**Step 4 - If they want changes.**

Fold their feedback into the brief and call `plan_story` again, changing the premise,
`must_include` or `avoid` - whichever their request affects.

Do not rewrite the outline yourself. An outline you edited by hand has not been reviewed
by anything, and it goes straight into the finished book. If the user wants something
different, change the brief and call `plan_story` again so the change is planned
properly.

Repeat steps 3 and 4 until the user approves. Whichever outline they approve is the one
you pass to `write_story`.

**Step 5 - Write the book.**

Only once the user has agreed, call `write_story` with the brief **and the outline
`plan_story` returned**. Pass that outline through exactly as you received it - do not
edit it, summarise it, or write one of your own. It is what the user approved, and the
book is built from it.

It makes several model calls and takes noticeably longer than `plan_story`.

**If a tool fails**

Report the error message as written, say that you are stopping, and ask the user how to
proceed. Do not retry a failed call with the same input.
"""
