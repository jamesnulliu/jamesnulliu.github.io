---
title: "Reinforcement Learning for LLMs"
date: 2025-06-02T08:25:00+08:00
lastmod: 2025-11-18T18:45:00+08:00
draft: false
author: ["jamesnulliu"]
keywords: 
    - reinforcement learning
    - llm
    - ppo
    - grpo
categories:
    - deeplearning
tags:
    - rl4llm
description: 
summary: 
comments: true
images: 
cover:
    image: ""
    caption: ""
    alt: ""
    relative: true
    hidden: true
---

## 1. Basics

- RLHF: Reinforcement Learning from Human Feedback
- SFT: Supervised Fine-Tuning

RL trains neural networks through **trial** and **error**. When finetuning a language model with RLHF, the model produces some text then receives a score/reward from a human annotator that captures the quality of that text. Then, we use RL to finetune the language model **to generate outputs with high scores**.

In this case, we cannot apply a loss function that trains the language model to maximize human preferences with supervised learning. This is because there’s no easy way to explain the score human give or connect it mathematically to the output of the neural network. In other words, we cannot backpropagate a loss applied to this score through the rest of the neural network. This would require that we are able to **differentiate (i.e., compute the gradient of) the system that generates the score**, which is a human that subjectively evaluates the generated text.

{{<image
src="/imgs/blogs/reinforcement-learning-for-llms/rl-structure.webp"
width=90%
caption=`The agent acts and receives rewards (and new states) from the environment. @Cameron R. Wolfe.`
>}}

Problems that are solved via RL tend to be structured in a similar format. Namely, we have an **agent** that is interacting with an **environment**; see the figure above. The agent has a state in the environment and produces actions, which can modify the current state, as output. As the agent interacts with the environment, it can receive both positive and negative rewards for its actions. **The agent’s goal is to maximize the rewards that it receives, but there is not a reward associated with every action taken by the agent**. Rather, rewards may have a long horizon, meaning that it takes several correct, consecutive actions to generate any positive reward.

## 2. MDP: Markov Decision Process

Markov Decision Process (MDP) is a way to formulate the system described above more formally and mathematically. Within an MDP, we have states, actions, rewards, transitions, and a policy, as shown in the equation below:

$$
\begin{cases}
s \in S & \text{State} \\
a \in A & \text{Action} \\
r_s \in \mathbb{R} & \text{Reward} \\
\pi(a|s) & \text{Policy} \\
T(s_{t+1}|s_{t},a_{t}) & \text{Transition function} \\
\end{cases}
$$

States and actions have discrete values, while rewards are real numbers. 

In an MDP, we define two types of functions: transition and policy functions. The policy takes a state as input, then outputs a probability distribution over possible actions. 

> Notably, the action that is chosen only depends on the current state and not any state history that precedes it. This is a key property of an MDP, which make the assumption that the next action only depends upon the current state.

Given this output, we can make a decision for the action to be taken from a current state, and the transition is then a function that outputs the next state based upon the prior state and chosen action. Using these components, the agent can interact with the environment in an iterative fashion, as the figure shown below.

{{<image
src="/imgs/blogs/reinforcement-learning-for-llms/rl-structure.webp"
width=90%
caption=`Structure of an MDP. @Cameron R. Wolfe.`
>}}

## References

- {{<href text="Basics of Reinforcement Learning for LLMs" url="https://cameronrwolfe.substack.com/p/basics-of-reinforcement-learning">}}