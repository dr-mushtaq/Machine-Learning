# Course Outline: Optimization Principles for Machine Learning (Revised)

**Revision notes:** Convex Optimization moved earlier (prerequisite for understanding non-convex deep learning landscapes); classical ML applications (logistic regression, SVMs) resequenced before deep-learning-specific material; Module 23 split into classical vs. modern advanced methods; duplicate bullets consolidated; automatic differentiation added to Module 2; cross-references added where content recurs.

---

## Module 1: Introduction to Optimization in Machine Learning

**Learning Objectives:** Understand what optimization means and why it is essential for training machine learning models.

**Key Concepts:**
- Definition of optimization
- Optimization problems in machine learning
- Objective functions
- Decision variables
- Constraints
- Minimization vs. maximization
- Role of optimization in model training
- Training loss vs. validation loss

---

## Module 2: Mathematical Foundations for Optimization

**Learning Objectives:** Build the mathematical knowledge required to understand optimization algorithms.

**Key Concepts:**
- Functions and variables
- Vectors and matrices
- Linear algebra basics
- Norms and distances
- Derivatives, partial derivatives, gradients
- Jacobian matrix, Hessian matrix
- Taylor series approximation
- Automatic differentiation and computational graphs *(new — foundation for backpropagation in Module 20)*

---

## Module 3: Loss and Objective Functions

**Learning Objectives:** Understand how machine learning problems are represented as mathematical optimization problems.

**Key Concepts:**
- Loss functions, cost functions
- Empirical risk minimization
- Mean Squared Error, Mean Absolute Error
- Binary Cross-Entropy, Categorical Cross-Entropy
- Hinge loss *(introduced here; revisited in depth in Module 12 for SVMs)*
- Choosing an appropriate loss function

*Regularized objective functions moved to Module 14, where regularization is covered in full.*

---

## Module 4: Fundamentals of Unconstrained Optimization

**Learning Objectives:** Learn how to find optimal solutions when no explicit constraints are applied.

**Key Concepts:**
- Unconstrained optimization
- Local minimum, global minimum
- Stationary points
- First- and second-order optimality conditions
- Convex and non-convex functions
- Saddle points *(revisited in Module 16 in the context of high-dimensional deep learning)*
- Optimization landscapes

---

## Module 5: Convex Optimization

**Learning Objectives:** Develop an understanding of convex optimization and why it is mathematically attractive, before studying algorithms that exploit or contend with (non-)convexity.

**Key Concepts:**
- Convex sets, convex functions
- Strict convexity, strong convexity
- Convex optimization problems
- Global optimality
- Jensen's inequality
- Convex loss functions
- Applications in machine learning

---

## Module 6: Gradient Descent

**Learning Objectives:** Understand the most fundamental optimization algorithm used in machine learning.

**Key Concepts:**
- Gradient direction
- Gradient Descent algorithm
- Learning rate, step size
- Convergence
- Initialization
- Learning-rate selection
- Gradient Descent implementation
- Common convergence problems

---

## Module 7: Batch, Stochastic, and Mini-Batch Gradient Descent

**Learning Objectives:** Compare different approaches for calculating gradients and understand their practical use.

**Key Concepts:**
- Batch, Stochastic, and Mini-Batch Gradient Descent
- Computational efficiency
- Gradient noise
- Batch size
- Convergence behavior
- Advantages and disadvantages of each method

---

## Module 8: Learning Rate and Optimization Dynamics

**Learning Objectives:** Understand how the learning rate affects training performance and convergence.

**Key Concepts:**
- Small vs. large learning rates
- Learning-rate schedules: step decay, exponential decay, cosine annealing
- Warm-up
- Cyclical learning rates
- Adaptive learning rates
- Learning-rate tuning

---

## Module 9: Momentum-Based Optimization

**Learning Objectives:** Learn how momentum improves Gradient Descent and accelerates convergence.

**Key Concepts:**
- Limitations of standard Gradient Descent
- Momentum, velocity, momentum coefficient
- Exponentially weighted averages
- Nesterov Accelerated Gradient
- Oscillation reduction
- Faster convergence

---

## Module 10: Adaptive Optimization Algorithms

**Learning Objectives:** Understand modern optimizers commonly used in deep learning.

**Key Concepts:**
- AdaGrad, RMSProp
- Adam, AdamW, AMSGrad
- Adaptive learning rates
- First and second moments, bias correction
- Comparison of optimizers
- Choosing an optimizer

---

## Module 11: Optimization for Logistic Regression and Linear Models

**Learning Objectives:** Apply optimization methods to classical machine learning algorithms, before moving into deep-learning-specific challenges.

**Key Concepts:**
- Linear regression optimization, least squares
- Logistic regression
- Maximum likelihood estimation
- Gradient-based optimization
- Regularized regression
- Closed-form vs. iterative solutions

---

## Module 12: Optimization in Support Vector Machines

**Learning Objectives:** Understand optimization from the perspective of margin-based machine learning.

**Key Concepts:**
- Maximum-margin classification
- Hinge loss (full treatment)
- Hard-margin SVM, soft-margin SVM
- Primal formulation, dual formulation
- Lagrange multipliers
- Kernel optimization

---

## Module 13: Constrained Optimization

**Learning Objectives:** Learn how optimization problems are solved when constraints are present.

**Key Concepts:**
- Equality constraints, inequality constraints
- Feasible regions
- Lagrange multipliers, Lagrangian function
- Karush-Kuhn-Tucker conditions
- Primal and dual problems, duality
- Constrained machine learning problems

---

## Module 14: Regularization as Optimization

**Learning Objectives:** Understand how regularization controls model complexity and improves generalization.

**Key Concepts:**
- Overfitting, bias-variance trade-off
- L1 regularization, L2 regularization, Elastic Net
- Weight decay, sparsity
- Regularized objective functions
- Regularization in neural networks
- Early stopping as an implicit regularizer *(cross-reference: revisited as a search-control technique in Module 22)*

---

## Module 15: Second-Order Optimization Methods

**Learning Objectives:** Explore optimization algorithms that use curvature information.

**Key Concepts:**
- Second derivatives, Hessian matrix
- Newton's Method, Newton-Raphson method
- Quasi-Newton methods: BFGS, L-BFGS
- Computational limitations
- First-order vs. second-order methods

---

## Module 16: Optimization Challenges in Deep Learning

**Learning Objectives:** Understand why optimizing deep neural networks is difficult.

**Key Concepts:**
- Non-convex optimization
- High-dimensional parameter spaces
- Vanishing and exploding gradients
- Saddle points in high dimensions *(builds on Module 4)*
- Flat and sharp minima
- Poor initialization
- Ill-conditioning, gradient instability

---

## Module 17: Weight Initialization and Gradient Flow

**Learning Objectives:** Understand how parameter initialization influences neural network optimization.

**Key Concepts:**
- Random and zero initialization
- Xavier/Glorot initialization, He initialization
- Initialization and activation functions
- Gradient propagation
- Stable training
- Initialization best practices

---

## Module 18: Normalization and Optimization

**Learning Objectives:** Learn how normalization techniques improve stability and training speed.

**Key Concepts:**
- Feature normalization, standardization
- Batch Normalization, Layer Normalization, Group Normalization
- Internal activation distributions
- Optimization stability
- Normalization in Transformers

---

## Module 19: Gradient Clipping and Training Stability

**Learning Objectives:** Learn methods for preventing unstable gradient updates.

**Key Concepts:**
- Exploding gradients
- Gradient clipping by value, gradient clipping by norm
- Gradient monitoring
- Training stability in RNNs and Transformers

---

## Module 20: Optimization for Neural Networks

**Learning Objectives:** Connect optimization principles directly to neural network training.

**Key Concepts:**
- Forward propagation
- Loss computation
- Backpropagation and the chain rule (using autodiff from Module 2)
- Gradient calculation, parameter updates
- Epochs and iterations
- Mini-batch training
- Neural network convergence

---

## Module 21: Optimization for CNNs, RNNs, and Transformers

**Learning Objectives:** Explore optimization considerations for different deep learning architectures.

**Key Concepts:**
- CNN optimization, RNN optimization
- LSTM and GRU training
- Transformer optimization, AdamW in Transformers
- Learning-rate warm-up (recap from Module 8)
- Gradient clipping (recap from Module 19)
- Large-batch training

---

## Module 22: Hyperparameter Optimization

**Learning Objectives:** Understand how optimization is used to select the best model configuration.

**Key Concepts:**
- Hyperparameters vs. parameters
- Grid Search, Random Search
- Bayesian Optimization, Hyperband, Optuna
- Early stopping as a search-control technique *(cross-reference: Module 14)*
- Search spaces, evaluation metrics

---

## Module 23: Advanced Optimization Techniques I — Classical Methods

**Learning Objectives:** Study foundational advanced optimization methods with roots in classical numerical optimization.

**Key Concepts:**
- Coordinate Descent
- Proximal Gradient Methods
- Mirror Descent
- Natural Gradient Descent
- Trust-region methods

---

## Module 24: Advanced Optimization Techniques II — Modern Deep Learning Methods

**Learning Objectives:** Study advanced methods developed specifically to improve deep learning training efficiency and generalization.

**Key Concepts:**
- Gradient accumulation
- Lookahead optimizer
- Sharpness-Aware Minimization (SAM)
- Large-scale optimization considerations

---

## Module 25: Distributed and Large-Scale Optimization

**Learning Objectives:** Understand optimization when models and datasets become very large.

**Key Concepts:**
- Data parallelism, model parallelism
- Distributed Gradient Descent
- Synchronous vs. asynchronous training
- Gradient accumulation at scale
- Large-batch optimization
- Communication overhead
- Distributed deep learning

---

## Module 26: Diagnosing Optimization Problems

**Learning Objectives:** Develop practical skills for identifying and solving training problems.

**Key Concepts:**
- Loss curves
- Training plateaus, diverging loss, oscillating loss
- Overfitting vs. optimization failure
- Gradient monitoring
- Learning-rate debugging
- Optimizer selection, batch-size adjustment
- Training diagnostics

---

## Module 27: Practical Optimization with Python

**Learning Objectives:** Implement and experiment with optimization algorithms using Python.

*Instructor note: consider threading short hands-on exercises from this module throughout Modules 6–22, rather than concentrating all implementation work here — 25 modules of theory before any code risks disengagement.*

**Key Concepts:**
- NumPy implementation
- Scikit-learn optimization
- PyTorch optimizers, TensorFlow/Keras optimizers
- Implementing Gradient Descent from scratch
- Comparing SGD, Adam, and AdamW
- Plotting loss curves
- Hyperparameter experiments, performance evaluation

---

## Module 28: Optimization for Large Language Models (Practical Mechanics)

**Learning Objectives:** Understand the practical optimization techniques behind training and fine-tuning modern large-scale language models.

**Key Concepts:**
- Transformer training
- AdamW, weight decay
- Learning-rate warm-up, cosine decay
- Gradient accumulation, mixed-precision training, gradient checkpointing
- Fine-tuning optimization
- Parameter-efficient fine-tuning, LoRA optimization

---

## Module 29: Research-Level Optimization Topics (Theory)

**Learning Objectives:** Explore current and advanced research directions in machine learning optimization, building on the practical LLM mechanics from Module 28.

**Key Concepts:**
- Optimization landscapes
- Generalization and flat minima, sharpness
- Neural Tangent Kernel
- Implicit regularization
- Optimization-generalization relationship
- Scaling laws
- Adaptive optimizer research
- Efficient fine-tuning and optimization of foundation models

---

## Module 30: Case Studies and Comparative Experiments

**Learning Objectives:** Apply different optimization strategies and compare their effectiveness.

**Key Concepts:**
- SGD vs. Adam, Adam vs. AdamW
- Effect of learning rate, batch size, momentum, initialization, normalization
- Optimizer benchmarking
- Interpreting convergence graphs

---

## Module 31: Final Project — End-to-End Optimization of a Machine Learning Model

**Learning Objectives:** Combine all optimization concepts in a complete practical project.

**Project Steps:**
1. Select a machine learning or deep learning problem.
2. Prepare and normalize the dataset.
3. Define the objective and loss function.
4. Build the model.
5. Select an optimizer.
6. Configure the learning rate.
7. Experiment with batch sizes.
8. Apply regularization.
9. Test learning-rate schedules.
10. Compare multiple optimizers.
11. Monitor gradients and loss curves.
12. Diagnose optimization problems.
13. Perform hyperparameter optimization.
14. Evaluate model performance.
15. Present and justify the final optimization strategy.

---

## Learning Progression

**Beginner (Modules 1–9):** Optimization fundamentals, mathematics (including autodiff), loss functions, convexity, and Gradient Descent variants.

**Intermediate (Modules 10–15):** Adaptive optimizers, classical ML applications (linear/logistic regression, SVMs), constrained optimization, regularization, and second-order methods.

**Advanced (Modules 16–22):** Deep learning optimization challenges, initialization, normalization, gradient stability, architecture-specific optimization, and hyperparameter search.

**Expert/Research Level (Modules 23–31):** Advanced classical and modern optimization methods, distributed training, diagnostics, hands-on Python implementation, LLM optimization, research topics, comparative experiments, and an end-to-end project.
