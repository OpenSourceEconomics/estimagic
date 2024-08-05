(ep-00)=

# EP-00: Governance model & code of conduct

```{eval-rst}
+------------+------------------------------------------------------------------+
| Author     | `Maximilian Blesch <https://github.com/MaxBlesch>`_,             |
|            | `Janoś Gabler <https://github.com/janosg>`_,                     |
|            | `Hans-Martin von Gaudecker <https://github.com/hmgaudecker>`_,   |
|            | `Annica Gehlen <https://github.com/amageh>`_,                    |
|            | `Sebastian Gsell <https://github.com/segsell>`_,                 |
|            | `Tim Mensinger <https://github.com/timmens>`_,                   |
|            | `Mariam Petrosyan <https://github.com/mpetrosian>`_,             |
|            | `Tobias Raabe <https://github.com/tobiasraabe>`_,                |
|            | `Klara Röhrl <https://github.com/roecla>`_                       |
+------------+------------------------------------------------------------------+
| Status     | Accepted                                                         |
+------------+------------------------------------------------------------------+
| Type       | Standards Track                                                  |
+------------+------------------------------------------------------------------+
| Created    | 2022-04-28                                                       |
+------------+------------------------------------------------------------------+
| Resolution |                                                                  |
+------------+------------------------------------------------------------------+
```

## Purpose

This document formalizes the optimagic code of conduct and governance model. In case of
changes, this document can be updated following the optimagic Enhancement Proposal
process detailed below.

```{include} ../../../CODE_OF_CONDUCT.md
```

## optimagic governance model

### Summary

The governance model strives to be lightweight and based on
[consensus](https://numpy.org/doc/stable/dev/governance/governance.html#consensus-based-decision-making-by-the-community)
of all interested parties. Most work happens in GitHub issues and pull requests (regular
decision process). Any interested party can voice their concerns or veto on proposed
changes. If this happens, the optimagic Enhancement Proposal (EP) process can be used to
iterate over proposals until consesus is reached (controversial decision process). If
necessary, members of the steering council can moderate heated debates and help to
broker a consensus.

### Regular decision process

Most changes to optimagic are additions of new functionality or strict improvements of
existing functionality. Such changes can be discussed in GitHub issues and discussions
and implemented in pull requests. They do not require an optimagic Enhancement Proposal.

Before starting to work on optimagic, contributors should read
[how to contribute](how-to) and the [styleguide](styleguide). They can also reach out to
existing contributors if any help is needed or anything remains unclear. We are all
happy to help onboarding new contributors in any way necessary. For example, we have
given introductions to git and GitHub in the past to help people make a contribution to
optimagic.

Pull requests should be opened as soon as work is started. They should contain a good
description of the planned work such that any interested party can participate in the
discussion around the changes. If planned changes turn out to be controversial, their
design should be discussed in an optimagic Enhancement Proposal before the actual work
starts. When the work is finished, the author of a pull request can request a review. In
most cases, previous discussions will show who is a suitable reviewer. If in doubt, tag
[janosg](https://github.com/janosg). Pull requests can be merged if there is at least
one approving review.

Reviewers should be polite, welcoming and helpful to the author of the pull request who
might have spent many hours working on the changes. Authors of pull requests should keep
in mind that reviewers' time is valuable. Major points should be discussed publicly on
GitHub, but very critical feedback or small details can be moved to private discussions
— if the latter are necessary at all (see
[the bottom section of this blog post](https://rgommers.github.io/2019/06/the-cost-of-an-open-source-contribution/)
for an excellent discussion of the burden that review comments place on maintainers,
which might not always be obvious). Video calls can help if a discussion gets stuck. The
code of conduct applies to all interactions related to code reviews.

### optimagic Enhancement Proposals (EPs) / Controversial decision process

Large changes to optimagic can be proposed in optimagic Enhancement Proposals, short
EPs. They serve the purpose of summarising discussions that may happen in chats, issues,
pull requests, in person, or by any other means. Simple extensions (like adding new
optimizers) do not need to be discussed with such a formal process.

EPs are written as markdown documents that become part of the documentation. Opening an
EP means opening a pull request that adds the markdown document to the documentation. It
is not necessary to already have a working implementations for the planned changes, even
though it might be a good idea to have rough prototypes for solutions to the most
challenging parts.

If the author of an EP feels that it is ready to be accepted they need to make a post in
the relevant [Zulip topic](https://ose.zulipchat.com) and a comment on the PR that
contains the following information:

1. Summary of all contentious aspects of the EP and how they have been resolved
1. Every interested party has seven days to comment on the PR proposing the EP, either
   with approval or objections. While only objections are relevant for the decision
   making process, approvals are a good way to signal interest in the planned change and
   recognize the work of the authors.
1. If there are no unresolved objections after seven days, the EP will automatically be
   accepted and can be merged.

Note that the pull requests that actually implement the proposed enhancements still
require a standard review cycle.

### Steering Council

The optimagic Steering Council consists of five people who take responsibility for the
future development of optimagic and the optimagic community. Being a member of the
steering council comes with no special rights. The main roles of the steering council
are:

- Facilitate the growth of optimagic and the optimagic community by organizing community
  events, identifying funding opportunities and improving the experience of all
  community members.
- Develop a roadmap, break down large changes into smaller projects and find
  contributors to work on the implementation of these projects.
- Ensure that new contributors are onboarded and assisted and that pull requests are
  reviewed in a timely fashion.
- Step in as moderators when discussions get heated, help to achieve consensus on
  controversial topics and enforce the code of conduct.

The Steering Council is elected by the optimagic community during a community meeting.

Candidates need to be active community members and can be nominated by other community
members or themselves until the start of the election. Nominated candidates need to
accept the nomination before the start of the election.

If there are only five candidates, the Steering Council is elected by acclamation. Else,
every participant casts five votes. The 5 candidates with the most votes become elected.
Candidates can vote for themselves. Ties are resolved by a second round of voting where
each participant casts as many votes as there are positions left. Remaining ties are
resolved by randomization.

Current memebers of the optimagic Steering Council are:

- [Janoś Gabler](https://github.com/janosg)
- [Annica Gehlen](https://github.com/amageh)
- [Hans-Martin von Gaudecker](https://github.com/hmgaudecker)
- [Tim Mensinger](https://github.com/timmens)
- [Mariam Petrosyan](https://github.com/mpetrosian)

### Community meeting

Community meetings can be held to elect a steering council, make changes to the
governance model or code of conduct, or to make other decisions that affect the
community as a whole. Moreover, they serve to keep the community updated about the
development of optimagic and get feedback.

Community meetings need to be announced via our public channels (e.g. the
[zulip workspace](https://ose.zulipchat.com) or GitHub discussions) with sufficient time
until the meeting. The definition of sufficient time will increase with the size of the
community.
