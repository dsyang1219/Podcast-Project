# Talking at the Audience — presentation script

Timed for about 18 minutes of talking plus questions. Bracketed stage directions are for you; everything else is speakable. Cut the marked *[optional]* passages to land at 12 minutes.

---

**[Slide 1 — Title]**

Thanks, everyone. I'm going to talk about political podcasts: how left- and right-leaning hosts talk to their listeners, and who those listeners turn out to be. The short version is three sentences. Left and right differ in how hosts address the audience, not in how hostile they are. People who get political news from podcasts are distinctively cynical about institutions. And the way a host talks to you predicts who you are. Everything else is evidence for those three claims.

**[Slide 2 — Motivation]**

Political podcasts are now one of the largest political news sources in the country. The biggest shows draw more people than cable news programs. And almost everything we know about partisan media comes from television and print: incivility, outrage, horse-race framing.

Podcasts are a different form. They're long, they're intimate, and for hours a week one voice is speaking directly to you. The intuitive story is that right-leaning shows are angrier. Nobody had actually measured that at scale. And nobody had connected what a host says to the people who actually listen.

So there are two gaps. The content gap: there was no transcribed, validated corpus of the field. The audience gap: even where people had studied podcast content, they couldn't link a specific show's language to that show's own listeners. The survey linkage is what makes this paper different.

**[Slide 3 — Research questions]**

Three questions. First, how do left- and right-leaning political podcasts differ in how hosts talk? Second, who listens to political podcasts, and how do they differ from other news consumers? Third, does a show's mode of address predict who its audience is and what they believe?

I'll flag now that the third question is where the work is exploratory. The first two are on firmer ground, and I'll be clear about the difference when we get there.

**[Slide 4 — Literature]** *[optional: 45 seconds]*

Three literatures make predictions here. Media malaise and the spiral of cynicism, from Robinson through Cappella and Jamieson and Ladd, predicts that news diets outside institutional journalism go with lower institutional trust. We find that. The incivility and outrage literature, Mutz and Reeves, Sobieraj and Berry, predicts that right-leaning shows are more hostile. We don't find that. And Horton and Wohl's para-social interaction, direct personal address that builds a one-sided relationship with a performer, fits what we do see: address, not hostility, is the partisan difference.

**[Slide 5 — The corpus]**

The data. We took the Apple Podcasts US Politics chart, ranks one through two hundred fifty, frozen on one date in July. A fixed-order filter brought that to two hundred four shows. We transcribed twenty-nine thousand episodes with faster-whisper on two GPU nodes, about a hundred times real time. That's one point three eight million passages of seven hundred fifty characters each.

One scope note. The frame is the Politics chart, so the largest political shows, Shapiro, The Daily, Tucker, are filed under other categories and aren't in the corpus. They come back later as an out-of-frame test.

**[Slide 6 — The surveys]**

Then two national surveys. The Kettering–Gallup Democracy for All study, about twenty thousand respondents, asked people to type in their top three news sources. Fifty-one thousand open-text answers. That's the key asset: people typed the names of shows, so we can match a show's transcripts to that show's own listeners. Seven hundred ninety-two listeners across fifty-six political podcasts, in the end.

And the Pew American Trends Panel, where podcast news use is measured in mid-2024 and institutional trust is measured two to eight months later. Pew has no show names, but it has time order.

**[Slide 7 — Measuring address]**

How do we measure how a host talks to the listener? Three components, each a rate per ten thousand words, averaged as z-scores. Syntactic imperatives, a bare verb heading a clause with no subject, from a dependency parse. Dictionary imperatives: look, listen, remember, understand. And second-person pronouns.

Here's what it sounds like. *[read the example on the slide]* "You need to understand what they're doing here. Look at what happened last week. Don't let them tell you otherwise."

Two things about the measure. Every instrument was published before this study; we wrote no word lists. And when we decompose it, the deontic second person, you need to, you should, is the strongest single component against host ideology. The filler "you know" is null. It's the second-person subject that carries the signal, not the modality.

**[Slide 8 — Labels]**

The ideology labels come from an LLM classifier at the passage level, and we validate them four ways. Against hosts' campaign contributions, r of point six eight. Against editorial and expert ratings, kappa around point seven five. And against the party identification of each show's own listeners in Kettering: r of point nine one, and the label predicts the audience's majority party in twenty-one of twenty-two shows. Text, money and listeners agree.

**[Slide 9 — Inference]** *[optional: 45 seconds; keep if the room is methods-heavy]*

One slide on inference, because the show-linked estimates have a real problem: exposure is assigned by show, and a few shows hold most of the listeners. Ordinary clustered standard errors are anti-conservative there. So every show-clustered estimate reports CR2 with Satterthwaite degrees of freedom as the primary p, and a wild cluster bootstrap alongside. The degrees of freedom, about nine on the full frame, is the honest count of how many clusters the coefficient is identified from. And the out-of-frame test was pre-registered and hashed before any audio was fetched.

**[Slide 10 — Finding 1]**

First finding. *[point to the chart]* Each bar is the correlation between a register measure and the host's donation-based ideology. The address measures sit at point three to point four. Insult is exactly zero. Profanity and hedging, if anything, go the other way.

Right-leaning hosts speak to the audience in the second person and issue more instructions. They are not more insulting. The between-side difference on the composite is d of point seven eight, format-controlled. Insult tracks ideological intensity, how extreme a show is, but not which direction. And a classifier on register features separates left from right at AUC point eight between shows and barely above chance within them. Register is a property of the show.

**[Slide 11 — Stable trait]**

And it doesn't move. We ran a within-show event study across eight political shocks from 2020 to 2025: two general elections, a midterm, the inauguration, the assassination attempt, Biden's withdrawal, January 6, Dobbs. Listener-directed address does not shift on either side. The 2024 election "shift" is indistinguishable from sixty placebo dates. It barely varies by topic either. Hosts don't ramp it up before elections. It's how they talk.

That matters for what comes next: if address were a reaction to the news, it would be a poor candidate for a stable audience-selection mechanism. Because it's a trait, a listener who chooses a show is choosing a mode of address that will be there every week.

**[Slide 12 — Finding 2]**

Second finding, and this one doesn't depend on the corpus at all. *[chart]* People who name a political podcast as a news source score about point two two standard deviations higher on an eight-item institutional cynicism scale than other news consumers, with a full control ladder: party, attention, age, education, income, race, gender, social media use, number of sources. The coefficient moves three percent across the whole ladder.

Read the bars top to bottom. Diets with no institutional source at all are about point two SD more cynical; podcast-only is the far end, point two seven. And then the within-group bars: the same in both parties, and identical at low and high education. The interaction with education is p of point nine one. So this is not "podcasts prey on the less educated." Education moves the baseline, not the podcast gap.

*[side panel]* The gradient is the mechanism-shaped fact: every additional mainstream source you name is worth minus point one zero SD. Podcasts are the far end of a non-institutional-diet gradient, not a category of their own.

**[Slide 13 — Pew]**

It replicates in Pew, with exposure measured first. *[chart]* Per step of podcast-news frequency in mid-2024, trust in national and local news organizations is lower months later, trust in social media is higher, and trust in friends and family is flat. That shape is the point. If podcast listeners were just generally distrustful people, the friends-and-family bars would drop too. A response-style artifact can't produce this pattern.

This is temporal order, not identification. People choose their podcasts. But it's the minimal condition, and it holds.

**[Slide 14 — Who they are]**

Who are these listeners? Not who you might expect. *[chart]* Podcast-only listeners are more civil-libertarian on the democratic-norms battery: less willing to have media take direction from government, less willing to expand presidential power, less willing to bar radicals from office, more willing to let them protest. They're no more strongly partisan, no lonelier, and no less registered. Engaged anti-institutionalism, not disengagement.

The contrast is platform-only diets, people who get their news only from social platforms. They show none of the civil-libertarian pattern and instead carry weaker national identity, lower registration and more tolerance of political violence. That's the disengaged profile. Podcast listeners are the opposite.

**[Slide 15 — Finding 3]**

Now the two halves meet. Third finding: listener-directed shows reach less formally educated audiences. *[chart]* Minus point two seven standard deviations of education per standard deviation of address, with party and show lean controlled, and it doesn't move when we add age, gender, income, race, urbanicity and attention.

It replicates in four samples, including the six out-of-frame shows, where nothing else replicated. It holds within left shows and within right shows. It's education specifically: income, age and gender are null. And people who name the show as their first source show it more sharply than people who name it third, which is the fingerprint of selection.

The way to think about this is register–audience fit. Direct, imperative, second-person speech is the involved, oral register, and it reaches people with less schooling regardless of party. This is about who a style reaches, not what it does to them.

**[Slide 16 — Para-social]** *[optional: 40 seconds]*

A bridge from Pew. Among regular consumers of news from influencers, the more often people get news from podcasts, the more likely they are to say they feel a personal connection to an influencer, that they follow them, that the influencer helped them understand. The relational and opinion items move. The informational items, basic facts, "their news is different," don't. That's the audience-side signature of direct, personal address.

**[Slide 17 — Finding 5, exploratory]**

Fifth finding, and I want to say the word before the numbers: this one is exploratory.

Within political podcasts, inside the corpus and among the largest political podcasts outside it, listener-directed address is associated with election distrust, plus point two seven SD per SD, and with institutional cynicism, plus point two two. *[chart]* The blue bars are the corpus, the red bars are the six big out-of-frame shows, and they agree. The grey bars are what remains once populist vocabulary and ideological intensity, the two rival content features, are in the same model. Election distrust and cynicism survive.

Why exploratory? We pre-registered this test on ten large chart-absent shows and it did not pass as frozen. Two of those ten turned out not to be political podcasts by content, three and five percent political passages, and once the population is defined by content the association is there and it's robust to the threshold and to dropping any show. But that definition came after the test. So the paper reports it as exploratory and pre-registers the content frame for Year 2 of the Kettering study.

**[Slide 18 — Pre-registration]**

Here's the full record. Three pre-registered tests, all frozen and hashed before their data were examined, and all three failed on their own criteria. Everything after them, the content frame, the education result, the comparison with populist content, was found by searching, and the paper says so and reports the search.

I put this slide in because a reviewer will find the hashed files, and I'd rather you heard it from me. The honest framing is: hypothesis generated in discovery, tested, not confirmed on the frozen frame, recovered on a content-defined frame, pre-registered for confirmation.

**[Slide 19 — Limitations]**

Limits, briefly. Everything on the audience side is an association under self-selection; the Pew lag gives time order, not identification. The population is political podcasts defined by content, and that definition is post hoc. Address is format-sensitive: in interview shows the second person is often aimed at the guest. The three content features are collinear, and with fifty-six clusters we can rank them only so far. And two validation steps are still open: human coding of passage labels, and human re-annotation of the ad-detection gold set.

**[Slide 20 — Takeaways]**

So, the three claims. How the sides differ: in how hosts address their audience, not in how hostile they are, and that's a stable trait of the show, validated against donations, experts and listeners' own party. Who listens: people who get political news from podcasts are about point two SD more cynical about institutions, in two panels, in both parties, at every education level, and they're civil-libertarian, not disengaged. And what address predicts: within political podcasts, the hosts who talk *at* the listener reach less-educated audiences, robustly, who distrust elections and institutions, exploratorily.

**[Slide 21 — Next steps]**

What's next. Before submission: the two human validation steps and the disclosure section. For confirmation: the Year 2 pre-registration with the content frame stated in advance and election distrust as the primary outcome. And the study that actually answers the causal question: a survey experiment, the same segment with and without second-person and imperative address, randomly assigned, with efficacy and election-trust items after. It's cheap, it uses the measure this paper validates, and it's the only design that can tell us whether address does anything to a listener rather than who it attracts.

**[Slide 22 — Thanks]**

Thank you. Happy to take questions.

---

## Likely questions, and short answers

**"Isn't the address measure just picking up interview shows, where 'you' is the guest?"**
It's a real boundary and we say so. The corpus is monologue and co-host dominated. The interrogative component of second-person, the part most likely aimed at a guest, is unrelated to host ideology, and partialling it out leaves the validation unchanged. Cross-format comparison of *levels* needs diarization, which the archived audio permits and is on the list.

**"How do you know the population finding isn't just selection?"**
We don't claim it isn't. It's an association under self-selection, stated as such. What we can say is that it survives a very long control ladder, holds within parties and within education levels, replicates with exposure measured months before outcome, and has a shape, institutional down, informal up, interpersonal flat, that a general-distrust or response-style story can't produce.

**"You changed the population definition after the pre-registered test failed."**
Yes. The criterion is symmetric, objective, and insensitive to its threshold, and a show that's three percent political isn't a political podcast by any reading. But it came after, so it's exploratory, and Year 2 pre-registers it. That's exactly why slide 18 exists.

**"Why not just say populism, or intensity, explains it?"**
Because we can't rule that in or out. The three content features inter-correlate at point four to point five. Within the political frame, address is the one whose association with election distrust and cynicism survives the other two in the same model, but with fifty-six clusters that's a ranking, not an adjudication. Year 2 carries all three.

**"What's the effect size in plain terms?"**
Point two two SD on cynicism is roughly the gap between a Democrat and a Republican on the same scale during this period. Point two seven SD of education per SD of address is about a third of the distance between "some college" and "bachelor's degree."

**"Why the LLM labels and not human coders?"**
Show-level labels are validated four ways, including against the party of real listeners, which is independent of the model. Passage-level labels haven't been human-coded yet; a 200-passage two-coder check precedes submission.

**"Is 204 shows enough?"**
For the register contrast, yes: the between-side d is point seven eight with show-level permutation p below ten to the minus five. For the audience linkage the binding constraint is listeners per show, not shows, which is why we report Satterthwaite degrees of freedom and why Year 2 matters.
