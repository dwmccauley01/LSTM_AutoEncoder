import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Parameters
num_points = 50000
amplitude = 20
frequency = 5
noise_level = 5
large_spike_prob = 0.0001
large_spike_magnitude = 20

# Generate time points
x = np.linspace(0, 1, num_points)

# Generate square wave
square_wave = amplitude * np.sign(np.sin(2 * np.pi * frequency * x))

# Add random noise spikes
noise_spikes = noise_level * (np.random.random(num_points) - 0.5)
square_wave_with_noise = square_wave + noise_spikes

# Add very large spikes
large_spikes = np.random.random(num_points) < large_spike_prob
square_wave_with_noise[large_spikes] += np.random.choice([-large_spike_magnitude, large_spike_magnitude], size=large_spikes.sum())

# Create a DataFrame
df = pd.DataFrame({
    'Time': x,
    'SquareWaveWithNoise': square_wave_with_noise
})

# Save to CSV
df.to_csv('square_wave_with_large_spikes.csv', index=False)

# Plot the square wave with noise and large spikes
plt.figure(figsize=(10, 5))
plt.plot(x, square_wave_with_noise, label='Square Wave with Noise and Large Spikes')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.title('Square Wave with Noise and Large Spikes')
plt.legend()
plt.grid(True)
plt.show()

print('Square wave with noise and large spikes saved to square_wave_with_large_spikes.csv and plotted.')
