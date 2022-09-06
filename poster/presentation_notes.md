# Minimal introduction

- Interested in communication in A. leptorhynchus (Gymnotiform, wave-type, South America)
- In the lab: Much work on chirps and rises (fast and slower frequency increases)

# Introduction

- Weakly electric fish actively produce electric field using modified muscle cells in their electric organ.
- Used for electrolocation (navigation, foraging) and **communication**. Pulse-type produce pulses, wave-type have a continuous discharge, never stops (straight area in spectrogram). Each fish has its own characteristic EOD$f$, can be used to distinguish individuals. Electric fields can be measured using electrodes and amplitifier. 


# Main question

- Field recordings: Chrips and rises, but sometimes also other, much longer, more diverse modulations.
- When looking at spectra: In some cases two fish do this at the same time, with the same modulation.
- This is new: There are reports of more gradual modulations, but not in synchrony.

## Questions

What are fish doing when they syncmod?

1. Are they close during modulations?
2. Could this be jamming avoidance or a signal?
3. Do their spatiotemporal behaviors (movement patterns) change with modulations?

# Methods

- Dataset recorded in Colombia 2016.
- Grid of electrodes, covered 3.5 x 3.5 meters.
- Comibining all electrodes gives us all communication signals of all fish on the grid.
- Triangulating between electrodes for estimated positions for each fish.
- Allows us to estimate where fish are and what they say at all times  - in a natural population - without tagging (in fact we don't even see the fish!)
- Dataset consists of continous recs over 2 weeks!
- To detect syncmod: Covariance based approach (want to know more about event detection?).
- To detect spatial interactions: Use proximmity, velocity and heading direction.


# What we found

## Some examples

- As expected: Diverse syncmod (spectra).
- Lasted up to 10 minutes, very long compared to known communication signals.
- Then looked at positions during syncmod, found they are close!
- What do they do? First looked at  of fishpos.
- Plot: Fish positions during two time ranges:
  1. Random time, fish are somewhere on the grid.
  2. Time at syncmod, fish are very close.
- Movements are visualised in videos, much easier to look at.

## Some results

- Lab suggests: Close fish shift their frequency to increase electrolocation capabities (**JAR**)
- $\Delta$ EOD$f$: Does it increase or decrease? Does not change much, if so, it decreases. Not JAR!
- In most cases, one initiates, i.e. starts first. Who changes EOD$f$ the most?
- Here results are not as clear, but seem to suggest initiators increase more.
- Distance: Estimated distribution of dyad distances for all, interactors, interactions. Fish are close during interactions, but distr has a long tail.
- On the videos: During events, fish are closer, sometimes faster, swim towards each other. How does this relate to the start of a syncmod.
- Most important figure: Time centered around onset of syncmod.
- Rasterplot: Interaction events.
- Blue line: KDE for 'interaction events'
- Interaction events increase **after** the onset of syncmod! This also explains long tail in distro: Fish approach each other after syncmod started!
- Indicates that modulations are not JAR or side-effect of interactions, but initiate interaction!

# Context

- First analysis that looked at this behavior
- Indicates that there might be more to communication than just chirps and rises.

