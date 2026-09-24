# H01 Cortical Column

Live simulation of 13 644 neurons from the H01 human cortical column dataset.

## What it is

The H01 dataset is a 1 mm³ fragment of human cortex, imaged at nanometre resolution by Harvard and Google. This project takes the reconstructed connectome — 13 644 neurons and 116 611 synapses — and runs it as a spiking neural network in Brian2, visualized with Pygame.

It is not the whole brain. It is one cortical column, taken out of context. But it is real data, real cells, real connections.

## Features

- 13 644 neurons from the H01 dataset
- 116 611 synaptic connections (chemical + gap junctions)
- Live voltage-based visualization — colour from dark blue (rest) to white (high activity)
- Dynamic background noise to keep the network alive
- Pause, reset counter, real-time stats

## Requirements

- Python 3.10+
- brian2
- pygame
- numpy

## Installation

```bash
pip install brian2 pygame numpy
