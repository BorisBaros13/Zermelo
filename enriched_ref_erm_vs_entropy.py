'''
Here we will take the training winds from our favourite trial,
and enrich the reference trajectories set using 4 classes of reference
control:
1) The initial zero-heading one
2) Constantly pointing towards the target
3) Constantly pointing a bit above
4) Constantly pointing a bit below

Ideally this should explore the space well enough for us 
to learn optimal headings 

Our favourite trial comes from 
./erm_vs_entropy/2025-09-26_sim2_of_3.pt

'''


