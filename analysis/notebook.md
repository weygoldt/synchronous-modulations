# Notes: Analysis of simultaneous slow EOD modulations

## Battleplan

**To do**
- [ ] Fix filter implementation: Take rate from time array instead of using an integer
- [ ] Make realtive angle function flexible to take an input angle and norm around that to extract chasing, schooling, etc.
- [ ] Try to find chasing events and rises, plot timeseries
- [ ] Write docstrings for gridtools
- [ ] Write gridtools documentation
- [ ] Write gridtools requirements
- [ ] Include preprocessing script in gridtools package
- [ ] Seperate interaction event detector in gridtools into 3 functions and add a fourth that combines them.
- [ ] Implement more flexibility into the lag given to the sliding window covariance.

## Are simultaneous modulations random?
To determine to which probability simultaneous modulations are observed if they are random, I need a detector that detects all modulations, not just those that are simultaneously observed in two fish.

### Distance during interactions
- Estimate PDF of distances at all times of all interacting individuals and compare to PDF of distances during interactions. Result: Fish are close during interactions.

### Frequency difference during interactions
- Estimate the overall pairwise frequency difference PDF and compare to PDF of interactors. Result: Interactors are close in frequency. 
- Peak is at ~ 20 Hz $df$ but it also happens when $df$ is larger -> Could be jamming avoidance but also something else.
- Frequency difference of interactors during interactions is insignificantly lower but porobably because event ranges always include baseline EOD differences as well.
- I need a way to better estimate the actual frequency difference during interactions that better reflects the truth.
- Maybe compare resting $df$ to event df in boxplot? Paired scatter plot?

### Prevalence of frequency modulations in close encounters with close-frequency fish
- Jamming should always take place when two fish with similar EODf approach each other. Hence, compute all of such encounters by:
1. Threshold distances based on PDF peak width
2. Threshold frequency differences based on PDF peak width
3. Compute proportion of close incounters in which detected events fall
4. Result: Proportion of counters that elicit events!

## Who starts the interaction?
- Jamming avoidance in *A. leptorynchius* can either be a fish with higher frequency increasing its EOD to evade a "frequency attack" from below, or a lower fish increasing its EODf beyond the jamming stimulus. To find out which is more prevalent:
1. Compute initiator - reactor difference between resting EODf. If negative, initiator had smaller EODf.
2. Plot PDF of initiator - reactor diff.
3. Compare frequency initiator with spatial initiator. Is the approaching animal also the animal starting the synchronous modulation?

## Prevalence in sex
- Estimate sex ratio of all fish in dataset
- Compute sex of interacting individuals
- Count female-female, male-male, male-female interactions and make barplots
- Normalize counts to population sex ratio.

## More ideas concerning grid recordings
- Fish are randomly distributed on the grid - are they randomly distributed on the grid **over time** as well? I.e., do they keep a certain distance to not jam each other or do they not care at all?
- If they are not, jamming is a problem!
- Also, what happens when they come close? How long do they stay close to each other? Are there individuals that interact more frequently with each other or longer? One could cound number of close encounters and durations and construct social network graphs for them. Then one could relate this back to the fish "territories" (e.g. KDE area where KDE position probability is 95%). Do males and females react more often? How does this change during mating? If we implement chirp detection on the grid and detect chirps during mating season, we can track who mates with whom.
- Think about ideas to incorporate experimental approaches to in situ grid recorigs. Shelter, food and artificial electric stimuli could be combined with grid recordings in the wild.

# Notes about grid field trip

## Dominance 

- Test if they attack electrodes with a rise playback. If they do so, this might be a potential stimulus to introduce into a wild population. If they do so, we can also ask the question of which parameters make a rise a rise. Amplitude in modulation? Temporal pattern? 

# General notes

- [ ] Code a 'gridexplorer': Plot positions and frequency traces next to each other and dynamically adjust plottet position traces based on selected window in the frequency traces. Preprocessing parameters could also be included here.