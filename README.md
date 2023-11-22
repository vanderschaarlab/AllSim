# AllSim: Systematic Simulation and Benchmarking of Repeated Resource Allocation Policies in Multi-User Systems with Varying Resources

Numerous real-world systems, ranging from healthcare to energy grids, involve
users competing for finite and potentially scarce resources. Designing policies
for repeated resource allocation in such real-world systems is challenging for
many reasons, including the changing nature of user types and their (possibly urgent)
need for resources. Researchers have developed numerous machine learning
solutions for determining repeated resource allocation policies in these challenging
settings. However, a key limitation has been the absence of good methods
and test-beds for benchmarking these policies; almost all resource allocation
policies are benchmarked in environments which are either completely synthetic
or do not allow any deviation from historical data. In this paper we introduce
AllSim, which is a benchmarking environment for realistically simulating the
impact and utility of policies for resource allocation in systems in which users
compete for such scarce resources. Building such a benchmarking environment
is challenging because it needs to successfully take into account the entire collective
of potential users and the impact a resource allocation policy has on all
the other users in the system. AllSim’s benchmarking environment is modular
(each component being parameterized individually), learnable (informed by
historical data), and customizable (adaptable to changing conditions). These,
when interacting with an allocation policy, produce a dataset of simulated outcomes
for evaluation and comparison of such policies. We believe AllSim is
an essential step towards a more systematic evaluation of policies for scarce
resource allocation compared to current approaches for benchmarking such methods.

## :rocket: Installation

The library can be installed from source, using
```bash
$ pip install .
```
 * Install the library with with additional libraries for testing and development
```bash
 pip install .[testing]
```

## Usage
Examples of usage are available in the `tutorial\` directory. These can be used to produce the results of the paper.

## Data

The data used in the tutorial and paper is private. It is available for members of the van der Schaar Lab. However, the tutorials should still be a useful reference if you want to use AllSim on your own data.
