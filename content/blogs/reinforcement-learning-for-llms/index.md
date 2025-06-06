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
width=85%
caption=`The agent acts and receives rewards (and new states) from the environment.`
>}}

Problems that are solved via RL tend to be structured in a similar format. Namely, we have an **agent** that is interacting with an **environment**; see the figure above. The agent has a state in the environment and produces actions, which can modify the current state, as output. As the agent interacts with the environment, it can receive both positive and negative rewards for its actions. **The agent’s goal is to maximize the rewards that it receives, but there is not a reward associated with every action taken by the agent**. Rather, rewards may have a long horizon, meaning that it takes several correct, consecutive actions to generate any positive reward.

## 2. Markov Decision Process (MDP)

### 2.1. Concepts and Definitions

Markov Decision Process (MDP) is a way to formulate the system described above more formally and mathematically. Within an MDP, we have **states**, **actions**, **rewards**, **transitions**, and a **policy**, as shown in the equation below:

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

In an MDP, we define two types of functions: **transition and policy functions**. The policy takes a state as input, then outputs a probability distribution over possible actions. 

> Notably, the action that is chosen only depends on the current state and not any state history that precedes it. This is a key property of an MDP, which make the assumption that the next action only depends upon the current state.

Given this output, we can make a decision for the action to be taken from a current state, and the transition is then a function that outputs the next state based upon the prior state and chosen action. Using these components, the agent can interact with the environment in an iterative fashion, as the figure shown below.

{{<image
src="/imgs/blogs/reinforcement-learning-for-llms/mdp-structure.webp"
width=80%
caption=`Structure of an MDP.`
>}}

- The policy describes how the agent chooses its next action given the current state. 
- The agent follows this strategy as it interacts with the environment. 
- The goal is to learn a policy that maximizes the reward that the agent receives from the environment.

As the agent interacts with the environment, we form a **trajectory** ($\tau$) of **states** ($s$) and **actions** ($a$) that are chosen throughout this process. Then, given the **reward** ($r_s$) associated with each of these states, we get a total return ($R(\tau)$) given by the equation below, where $\gamma$ is the discount factor:

$$
\begin{cases}
\tau &= \{s_0, a_0, s_1, a_1, \dots, s_t, a_t\} \quad &\text{(Trajectory)} \\ 
R(\tau) &= \sum_t \gamma^t r_{s_t} \quad &\text{(Return)}
\end{cases}
$$

$R(\tau)$ is the summed reward across the agent's full trajectory, but **rewards achieved at later time steps are exponentially discounted by the factor $\gamma$** --- This means that current rewards are more valuable than later rewards, due to both uncertainty and the simple fact that waiting to receive a reward is less desirable.

> TL;DR. The fact that the discount rate is bounded to be smaller than 1 is a mathematical trick to make an infinite sum finite. This helps proving the convergence of certain algorithms. See [Understanding the role of the discount factor in reinforcement learning](https://stats.stackexchange.com/questions/221402/understanding-the-role-of-the-discount-factor-in-reinforcement-learning).

**The goal of RL is to train an agent that maximizes this return**. As shown by the equation below, we can characterize this as finding a policy that maximizes the return over trajectories that are sampled from the final policy.

> Note that the policy is a probability distribution over actions at each time step given the current state, so the exact trajectory produced is not deterministic. Many different trajectories can be obtained depending upon how we sample actions from the policy. 

$$
\max_{\pi} ~ \mathbb{E}_{\tau \sim P_{\pi, T}} ~ R(\tau)
$$

where:

- $\max_{\pi}$ means to find the policy that yields the maximum return.
- $\mathbb{E}_{\tau \sim P_{\pi, T}}$ means to take the expectation or average over trajectories randomly sampled from a certain policy $\pi$ and transition function $T$.
- $R(\tau)$ is the return of the trajectory $\tau$.

### 2.2. A Classical Example

A classical example of an MDP is a maze, where the agent is trying to find the optimal path to the goal. 

{{<image
src="/imgs/blogs/reinforcement-learning-for-llms/agent-in-maze.webp"
width=70%
caption=`A simple traditional RL environment. The target is to train the agent to find the optimal (largest return) solution path.`
>}}

- **States**: The positions in the $2 \times 3$ maze. The positions can be represented as a one-hot vector.
- **Actions**: The possible moves (up, down, left, right) --- This is the agent's (i.e., the model's) each-step output.
- **Rewards**: The agent receives a reward of $+10$ for reaching the goal and $-10$ for reaching the trap. The agent receives a reward of $0$ for all other states.
- **Transition function**: The agent can move to an adjacent state based on the chosen action, but it cannot move through walls. The transition function defines how the agent moves from one state to another based on the action taken.
- **Policy**: The agent's policy is a probability distribution over the possible actions given the current state. For example, if the agent is in the state (0, 0), it might have a policy that gives a high probability to moving right and a low probability to moving down.
- **Trajectory**: The sequence of states and actions taken by the agent as it navigates the maze.


Like many problems that are solved with RL, this setup has an environment that is not differentiable (i.e., we can’t compute a gradient and train the model in a supervised fashion) and contains long-term dependencies, meaning that we might have to learn how to perform several sequential actions to get any reward. 

### 3. Taxonomy of modern RL algorithms

{{<image
src="/imgs/blogs/reinforcement-learning-for-llms/taxonomy-of-modern-rl-algorithms.webp"
width=100%
caption=`Taxonomy of modern RL algorithms.`
>}}


## 3. Deep Q-Leanring

### 3.1. Q-Learning

Q-Learning is a model-free RL algorithm, meaning that we don’t have to learn a model for the environment with which the agent interacts. The goal of Q-Learning is to **learn the value of any action at a particular state**. 

There are three key concepts to mention here:

1. Value $Q(s, a)$ corresponds to choosing action $a$ at current state $s$. The value not only contains the determined reward by taking action $a$ at state $s$, but also contains a **discounted** and **recursive** future Q-value of the next state $s'$ after taking action $max_a$ (i.e., the best action at state $s'$).
2. The higher a value $Q(s, a)$ is, the more valuable it is to take action $a$ at state $s$.
3. A look-up table (Q-table) must be maintained to store the Q values for each state-action pair. 

The algorithm first initialize all Q values as zero and pick an initial state with which to start the learning process. Then, iterate over the following steps:

- Pick an action to execute from the current state (using an $\varepsilon$-Greedy Policy).
- Get a reward and next state from the (model-free) environment.
- Update the Q value in the Q-table based on the Bellman equation.

Here we show a simplified update method which derives from the Bellman Optimality Equation and defines $Q(s_t, a_t)$ recursively:

$$
Q(s_t, a_t) = r_t + \gamma \max_{a} Q(s_{t+1}, a)
$$

where:
- $Q(s_t, a_t)$ is the Q value of the current state $s_t$ and action $a_t$.
- $r_t$ is the reward received after taking action $a_t$ at state $s_t$.
- $\gamma$ is the discount factor.
- $\max_{a} Q(s_{t+1}, a)$ is the maximum Q value of the next state $s_{t+1}$ over all possible actions $a$.

### 3.2. Deep Q-Learning (DQL)

In DQL, Q-table is replaced by a neural network.

{{<image
src="/imgs/blogs/reinforcement-learning-for-llms/dql-structure.webp"
width=90%
caption=`Structure of an DQL.`
>}}

 In DQL, we have two neural networks: the Q network and the target network. These networks are identical, but the exact architecture they use depend upon the problem being solved7. To train these networks, we first gather data by interacting with the environment. This data is gathered using the current Q network with an ε-greedy policy. This process of gathering interaction data for training the Q network is referred to as experience replay; see above.

From here, we use data that has been collected to train the Q network. During each training iteration, we sample a batch of data and pass it through both the Q network and the target network. The Q network takes the current state as input and predicts the Q value of the action that is taken (i.e., predicted Q value), while the target network takes the next state as input and predicts the Q value of the best action that can be taken from that state8 (i.e., target Q value). 

From here, we use the predicted Q value, the target Q value, and the observed reward to train the Q network with an MSE loss; see above. The target network is held fixed. Every several iterations, the weights of the Q network are copied to the target network, allowing this model to be updated as well. Then, we just repeat this process until the Q network converges. Notably, the dataset we obtain from experience replay is cumulative, meaning that we maintain all of the data we have observed from the environment throughout all iterations.

Why do we need the target network? The vanilla Q-learning framework leverages two Q values in its update rule: a (predicted) Q value for the current state-action pair and the (target) Q value of the best state-action pair for the next state. In DQL, we similarly have to generate both of these Q values. In theory, we could do this with a single neural network by making multiple passes through the Q network—one for the predicted Q value and one for the target Q value. However, the Q network’s weights are being updated at every training iteration, which would cause the target Q value to constantly fluctuate as the model is updated. To avoid this issue, we keep the target network separate and fixed, only updating its weights every several iterations to avoid creating a “moving target”. 

This idea of using a separate network to produce a training target for another network—referred to as knowledge distillation [6]—is heavily utilized within deep learning. Furthermore, the idea of avoiding too much fluctuation in the weights of the teacher/target model has been addressed in this domain. For example, the mean teacher approach [7] updates the weights of the teacher model as an exponential moving average of the student network’s weights; see above. In this way, we can ensure a stable target is provided by the teacher during training. 

## 4. Policy Gradients

In Policy Gradients, we will assume that our policy is a machine learning model (e.g., a deep neural network) with parameters $\theta$. This policy takes a state as input and predicts some distribution over the action space. We use this output to decide what action should be taken next within the MDP:

$$
\begin{cases}
s \in S & \text{State} \\
a \in A & \text{Action} \\
r_s \in \mathbb{R} & \text{Reward} \\
\pi_{\theta}(a|s) & \text{Policy} \\
T(s_{t+1}|s_{t},a_{t}) & \text{Transition function} \\
\end{cases}
$$

As our agent traverses the environment, it receives positive or negative reward signals for the actions it chooses and the states that it visits. Our goal is to learn a policy from these reward signals that maximizes total reward across an entire trajectory sampled from the policy. This idea is captured by the return, which sums the total rewards over an agent’s trajectory:

$$
\begin{cases}
\tau &= \{s_0, a_0, s_1, a_1, \dots, s_t, a_t\} \quad &\text{(Trajectory)} \\ 
R(\tau) &= \sum_t \gamma^t r_{s_t} \quad &\text{(Return)}
\end{cases}
$$

If $\gamma < 1$, then the return is **Infinite-Horizon Discounted Return**. If $\gamma = 1$, then the return is **Finite-Horizon Return**.

### 4.1. Value Functions and Advantage Functions

One final concept that will be especially relevant is that **value functions**. In RL, there are four basic value functions, all of which assume the infinite-horizon discounted return:

$$
\begin{align*}
V^{\pi}(s) &= \mathbb{E}_{\tau \sim (\pi,T)} [R(\tau)|s_0 = s] && \text{(On-Policy Value Function)} \\
Q^{\pi}(s, a) &= \mathbb{E}_{\tau \sim (\pi,T)} [R(\tau)|s_0 = s, a_0 = a] && \text{(On-Policy Action-Value Function)} \\
V^{*}(s) &= \max_{\pi} \mathbb{E}_{\tau \sim (\pi,T)} [R(\tau)|s_0 = s] && \text{(Optimal Value Function)} \\
Q^{*}(s, a) &= \max_{\pi} \mathbb{E}_{\tau \sim (\pi,T)} [R(\tau)|s_0 = s, a_0 = a] && \text{(Optimal Action-Value Function)}
\end{align*}
$$

- **On-Policy Value Functio**: expected return if starting in state $s$ and act according to policy $\pi$ afterwards.
- **On-Policy Action-Value Function**: expected return if you start in state $s$, take some action $a$ (may not come from the current policy), and act according to policy $\pi$ afterwards.
- **Optimal Value Function**: expected return if you start in state $s$ and always act according to the optimal policy afterwards.
- **Optimal Action-Value Function**: expected return if you start in state $s$, take some action a (may not come from the current policy), and act according to the optimal policy afterwards.

There is an important connection between the optimal policy in an environment and the optimal action-value function. Namely, the optimal policy selects the action in state $s$ that maximizes the value of the optimal action-value function.

Using the value functions described above, we can define a special type of function called an advantage function, which is heavily used in RL algorithms based on policy gradients: 

$$
A^{\pi}(s, a) = Q^{\pi}(s, a) - V^{\pi}(s) \quad \text{(Advantage Function)}
$$

- $Q^{\pi}(s, a)$ on-policy action-value function.
- $V^{\pi}(s)$ on-policy value function.

Simply put, the advantage function characterizes **how much better it is to take a certain action a relative to a randomly-selected action in state $s$ given a policy $\pi$**. Here, we should notice that the advantage function can be derived using the on-policy value and action-value functions defined before, as these functions assume that the agent acts according to a randomly-selected action from the policy $\pi$.

> "The value of your starting point is the reward you expect to get from being there, plus the value of wherever you land next." [3]

### 4.2. Policy Optimization

During the learning process, we aim to find parameters $\theta$ for our policy that maximize the objective function below:

$$
J(\pi_{\theta}) = \mathbb{E}_{\tau \sim (\pi_{\theta},T)} [R(\tau)]
$$

where:
- $\pi_{\theta}$ is the policy with network parameters $\theta$.
- $\tau$ is the trajectory sampled from the policy $\pi_{\theta}$ and transition function $T$.
- $T$ is the transition function that defines how the agent moves from one state to another based on the action taken.
- $R(\tau)$ is the return of the trajectory $\tau$.

In words, this objective function measures the expected return of trajectories sampled from our policy within the specified environment. 

If we want to find parameters $\theta$ that maximize this objective function, one of the most fundamental techniques that we can use is gradient ascent, which iterates over parameters $\theta$ using the update rule shown below:

$$
\theta_{t+1} = \theta_t + \alpha \nabla_{\theta} J(\pi_{\theta})|_{\theta_t}
$$

Do a lot of math to compute $\nabla_{\theta} J(\pi_{\theta})$, and the final result of the basic policy gradient is:

$$
\nabla_{\theta} J(\pi_{\theta}) = \mathbb{E}_{\tau \sim (\pi_{\theta}, T)} \left[ \sum_{t=0}^{T} \nabla_{\theta} \log \pi_{\theta}(a_t|s_t) R(\tau) \right]
$$

Now, we have an actual expression for the gradient of our objective function that we can use in gradient ascent! Plus, this expression only depends on the return of a trajectory and the gradient of the log probability of an action given our current state. As long as we instantiate our policy such that the gradient of action probabilities is computable (e.g., this is pretty easy to do if our policy is implemented as a neural network), we can easily derive both of these quantities.

### 4.3. Computing the Policy Gradient in Practice

In practice, we can estimate the value of this expectation by sampling a fixed number of trajectories, by:

- Sample several trajectories by letting the agent interact with the environment according to the current policy.
- Estimate the policy gradient using an average of relevant quantities over the fixed number of sample trajectories.

Then given a set of sampled trajectories $\mathcal{D} = \{\tau_0, \tau_1, \dots\}$, we can estimate the policy gradient $\overline{\nabla_{\theta} J(\pi_{\theta})}$ as follows:

$$
\overline{\nabla_{\theta} J(\pi_{\theta})} = \frac{1}{|\mathcal{D}|} \sum_{\tau \in \mathcal{D}} \sum_{t=0}^{T} \left[ \nabla_{\theta} \log \pi_{\theta}(a_t|s_t) R(\tau) \right]
$$

### 4.4. Variants of the Basic Policy Gradient

Given:

$$
\nabla_{\theta} J(\pi_{\theta}) = \mathbb{E} \left[ \sum_{t=0}^{T} \nabla_{\theta} \log \pi_{\theta}(a_t|s_t) \Psi_t \right]
$$

We have:

1. **Basic Policy Gradient**: 
   $$
   \Psi_t = R(\tau)
   $$
   Sum of all (potentially discounted) rewards obtained along the entire trajectory.
2. **Reward-to-Go**: 
   $$
   \Psi_t = \sum_{i=t}^{T} r_{s_i, a_i}
   $$
   Rewards after the current action.
3. **Reward-to-Go with Baseline**: 
   $$
   \Psi_t = \sum_{i=t}^{T} r_{s_i, a_i} - b(s_i)
   $$
   A baseline function to our expression that only depends on the current state.
4. **Vallina Policy Gradien**t:
   $$
   \Psi_t = A^{\pi_{\theta}}(s_t, a_t) = Q^{\pi_{\theta}}(s_t, a_t) - V^{\pi_{\theta}}(s_t)
   $$
   where $A^{\pi_{\theta}}(s_t, a_t)$ is the advantage function, $Q^{\pi_{\theta}}(s_t, a_t)$ is the action-value function, and $V^{\pi_{\theta}}(s_t)$ is the value function.

## 5. Proximal Policy Optimization (PPO)

## 6. Group Relative Policy Optimization (GRPO)


## References

[1] Cameron R. Wolfe. {{<href text="Basics of Reinforcement Learning for LLMs" url="https://cameronrwolfe.substack.com/p/basics-of-reinforcement-learning">}}.
[2] Cameron R. Wolfe. {{<href text="Proximal Policy Optimization (PPO): The Key to LLM Alignment" url="https://cameronrwolfe.substack.com/p/proximal-policy-optimization-ppo">}}
[3] Achiam, Josh. {{<href text="Spinning Up in Deep RL" url="https://spinningup.openai.com/en/latest/index.html">}}. OpenAI, 2018.