# Temporal features in Deep Learning for Network Intrusion Detection: MLP vs Transformer

## Overview

This repository contains the code and experiments from my bachelor's thesis, which compares two different deep learning architectures for network intrusion detection through multiclass classification.

## Motivation

The core research question explores whether incorporating new temporal features in network traffic data improves intrusion detection performance.

## Experiments

Both architectures are trained using supervised learning and evaluated on **three different datasets** representing diverse network scenarios and attack types. The models perform multiclass classification to distinguish between normal traffic and various types of intrusions.
The experimental results demonstrate both advantages and trade-offs of incorporating temporal features in intrusion detection tasks.

## Approach

The project compares two fundamentally different approaches:

- **MLP (Static Approach)**: Treats each network sample independently without considering temporal relationships between consecutive flows
- **Transformer (Temporal Approach)**: Leverages self-attention mechanisms to capture temporal correlations and sequential dependencies in network traffic

## Acknowledgments

I would like to thank my supervisor Stefano Iannucci and my co-supervisor Simone Albero for their support throughout this thesis, and especially Simone for his help with the technical aspects.