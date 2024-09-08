import matplotlib.pyplot as plt
from matplotlib.sankey import Sankey

# Create a Sankey diagram to represent the process flow
Sankey(flows=[1, 1, 1, 1, -1, -1, -1, -1],
       labels=['User Access', 'Mode Selection', 'Sign Language Input', 'Gesture Recognition',
               'Text Conversion', 'Language Translation', 'Text/Voice Output', 'Feedback Loop'],
       orientations=[0, 1, 0, 1, 0, -1, 0, -1],
       facecolor='blue').finish()

# Add a title
plt.title('Process Flow: Multi-Sign Language to Multi-Lingual Text Translation Platform')

# Show the plot
plt.show()
