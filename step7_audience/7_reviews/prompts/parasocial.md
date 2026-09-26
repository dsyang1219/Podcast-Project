# Review coding: parasocial bond and related dimensions

Sources. Parasocial interaction and relationship: Horton & Wohl (1956); the PSI scale of Rubin, Perse & Powell (1985)
(items: "I feel as if I know the host like a friend", "I look forward to hearing them", "I would miss them if the show ended",
"I feel sorry when they make a mistake", "the host makes me feel comfortable, as if with friends"); the PSI/PSR distinction of
Dibble, Hartmann & Rosaen (2016). Review-level dimensions of Funk, Lawrie & Speakman (2023): emotion, connection, praise,
loyalty. Each category below is a yes/no judgment about the REVIEW TEXT as written by the reviewer about the host(s) or show.

## System
You are a research assistant applying a published content-analysis codebook to podcast listener reviews. Apply the codebook
literally. Judge only the review shown. Return only the category names that apply.

## Task
Mark each category present in the REVIEW:

1  connection    the reviewer expresses a personal bond with the host(s): feeling that they know them, calling them a friend,
                 family, or company; the host "gets" them; they would miss the host; the host feels like someone they know
2  companionship the show accompanies the reviewer's daily life: commute, chores, "gets me through the day/week", "keeps me
                 sane", "part of my routine/morning"
3  direct_address the reviewer speaks TO the host(s) in the second person or by name ("thank you Ben", "keep it up guys",
                 "please have X back on")
4  emotion       strong expressed emotion about the host or show (love, joy, gratitude, comfort, anger, grief), beyond mild
                 approval
5  loyalty       long-term or exclusive listening: "listened since the beginning", "never miss an episode", "my number one",
                 "the only one I trust", "every day"
6  praise_person praise directed at the host as a PERSON (smart, funny, brave, honest, authentic, kind)
7  praise_content praise directed at the CONTENT (informative, well researched, balanced, good guests, production)
8  agreement     the reviewer endorses the show's political views or says it "tells it like it is", "speaks for me", "the truth"
9  anti_media    the reviewer disparages mainstream or legacy media or contrasts the show favourably with it
10 hostility     the reviewer attacks the host(s) or show (insults, contempt, accusations), whether or not politically framed
11 disagreement  the reviewer objects to the show's political views or bias without personal attack

Rules. A review may carry several categories or none. Praise of the host as a person (6) is not the same as connection (1):
"Ben is brilliant" is praise; "Ben feels like a friend" is connection. A one-star review can still carry connection or loyalty
("I have listened for years but..."). Sarcasm counts for its literal target only when clear.

REVIEW (rating {RATING} of 5, title: {TITLE}):
{TEXT}

Return the category names present, from this list only: connection, companionship, direct_address, emotion, loyalty,
praise_person, praise_content, agreement, anti_media, hostility, disagreement. If none apply, return an empty list.
