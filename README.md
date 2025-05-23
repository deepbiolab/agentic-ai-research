# Agentic AI Research

![](./assets/arch.svg)

Exploring agentic AI systems: research, development, and insights.

## Tutorials

All tutorials are available in the `./tutorials` directory. This tutorials are designed to explain the main idea behind of [References](#references) papers.


### Fundamental Components of Agentic Framework
A series of tutorials demonstrating how to build an agentic AI framework from scratch:

1. **[00-basic-llm-calls-and-agent](./tutorials/fundamental-agentic-framework/00-basic-llm-calls-and-agent.ipynb)**: Introduction to basic LLM API interactions and building a simple Agent class
   - Setting up OpenAI client
   - Understanding key parameters
   - Building a flexible Agent class
   - Example use cases with different agent roles

2. **[01-manage-memory](./tutorials/fundamental-agentic-framework/01-manage-memory.ipynb)**: Implementation of short-term memory management for AI agents
   - Simple memory implementation using lists
   - Creating a robust Memory class
   - Integration with Agent class

3. **[02-function-calling](./tutorials/fundamental-agentic-framework/02-function-calling.ipynb)**: Adding function calling capabilities to agents
   - Implementing tool calling functionality
   - Memory management with tool calls
   - Project management example with external data interactions

4. **[03-react-prompt-technique](./tutorials/fundamental-agentic-framework/03-react-prompt-technique.ipynb)**: ReACT (Reasoning + Acting) prompting implementation
   - Structured prompting for reasoning and action
   - Interactive wellness agent example
   - Step-by-step thought process demonstration

5. **[04-react-agent-from-scratch](./tutorials/fundamental-agentic-framework/04-react-agent-from-scratch.ipynb)**: Complete ReACT agent implementation
   - Memory layer implementation
   - Tool layer with function calling
   - ReACT loop implementation
   - Combine Self-reflection into ReACT agent

6. **[05-multi-agents-with-react](./tutorials/fundamental-agentic-framework/05-multi-agents-with-react.ipynb)**: Advanced implementation with multiple agents
   - Peer agent communication
   - Multi-agent task coordination
   - Complex problem-solving with agent collaboration

The tutorials demonstrate building increasingly sophisticated AI agents, from basic API calls to complex multi-agent systems with memory, function calling, and reasoning capabilities.

### Agentic Workflows with LangGraph

1. **[00-langchain-basics](./tutorials/agentic-workflows-with-langgraph/00-langchain-bascis.ipynb)**: Introduction to LangChain and its basic components
   - Setting up LangChain
   - Understanding key components
   - Building a simple agent with LangChain

## References

This curated list of papers mainly focus on the field of agentic AI in Scientific Discovery. It includes papers that explore the use of AI in scientific research, as well as papers that propose new methods and frameworks for scientific discovery. The list is organized by topic, and includes papers from a variety of sources, including academic journals, conferences, and preprint servers. The list is intended to be a resource for researchers and practitioners interested in the field of agentic AI in Scientific Discovery.

- [Highlights](#highlights)
- [Implementation Tools](#implementation-tools)
- [Datasets and Benchmarks](#datasets-and-benchmarks)
- [Scientific Discovery Applications](#scientific-discovery-applications)
- [Challenges and Open Problems](#challenges-and-open-problems)


#### Highlights

- [The AI Scientist: Towards fully automated open-ended scientific discovery](https://arxiv.org/pdf/2408.06292)  
  *Focus: Automating scientific discovery processes.*
- [ResearchAgent: Iterative research idea generation over scientific literature with large language models](https://arxiv.org/pdf/2404.07738)  
  *Focus: Generating research ideas from literature using LLMs.*
- [MemGPT: Towards LLMs as operating systems](https://arxiv.org/pdf/2310.08560)  
  *Focus: Treating LLMs as operating systems for task management.*
- [MLR-Copilot: Autonomous machine learning research based on large language models agents](https://arxiv.org/pdf/2408.14033)  
    *Focus: LLM agents for autonomous machine learning research.*
- [SciAgents: Automating scientific discovery through multi-agent intelligent graph reasoning](https://arxiv.org/pdf/2409.05556)  
  *Focus: Multi-agent reasoning for scientific discovery using graphs.*
- [ChatDev: Communicative agents for software development](https://arxiv.org/pdf/2307.07924)  
  *Focus: Collaborative agents for software engineering tasks.*
- [AutoGen: Enabling next-gen LLM applications via multiagent conversation framework](https://arxiv.org/pdf/2308.08155)  
  *Focus: Multi-agent conversation framework for advanced LLM applications.*
- [Living Software Systems with Generative & Agentic AI](https://arxiv.org/pdf/2408.01768)  
  *Focus: Building generative and agentic AI-driven software systems.*
- [Plan4MC: Skill reinforcement learning and planning for open-world long-horizon tasks](https://arxiv.org/pdf/2303.16563)  
  *Focus: Reinforcement learning and planning for long-horizon tasks.*
- [Agent Laboratory: Using llm agents as research assistants.](https://arxiv.org/pdf/2501.04227) 
    *Focus: Accepts human-provided research ideas and autonomously progresses through literature review, experimentation, and report writing.*

#### Implementation Tools

- [AutoGen: Enabling next-gen LLM applications via multiagent conversation framework](https://arxiv.org/pdf/2308.08155)  
  *Focus: Framework for building multi-agent LLM applications.*
- [MemGPT: Towards LLMs as operating systems](https://arxiv.org/pdf/2310.08560)  
  *Focus: LLM as an operating system for complex task workflows.*
- [CAMEL: Communicative Agents for 'Mind' Exploration of Large Language Model Society](https://arxiv.org/pdf/2303.17760)  
  *Focus: Exploring communication in multi-agent LLM societies.*
- [AutoGPT for Online Decision Making: Benchmarks and Additional Opinions](https://arxiv.org/pdf/2306.02224)  
  *Focus: Using AutoGPT for online decision-making tasks.*
- [MetaGPT: Meta Programming for a Multi-Agent Collaborative Framework](https://openreview.net/pdf?id=VtmBAGCN7o)  
  *Focus: Meta-programming for multi-agent collaboration.*



#### Datasets and Benchmarks

- [ResearchArena: Benchmarking LLMs’ ability to collect and organize information as research agents](https://arxiv.org/pdf/2406.10291)  
  *Focus: Benchmarking LLMs for research organization tasks.*
- [LAB-Bench: Measuring capabilities of language models for biology research](https://arxiv.org/pdf/2407.10362)  
  *Focus: Biological research benchmarks for language models.*
- [MatText: Do language models need more than text & scale for materials modeling?](https://arxiv.org/pdf/2406.17295)  
  *Focus: Evaluating LLMs for materials science tasks.*
- [MatSci-NLP: Evaluating scientific language models on materials science language tasks](https://arxiv.org/pdf/2305.08264)  
  *Focus: NLP benchmarks for materials science research.*
- [CiteME: Can language models accurately cite scientific claims?](https://arxiv.org/pdf/2407.12861)  
  *Focus: Evaluating LLMs' ability to cite scientific claims.*



#### Scientific Discovery Applications

- [LLaMP: Large Language Model Made Powerful for High-Fidelity Materials Knowledge Retrieval and Distillation](https://arxiv.org/pdf/2401.17244)  
  *Focus: High-fidelity materials knowledge retrieval with LLMs.*
- [Organa: A Robotic Assistant for Automated Chemistry Experimentation and Characterization](https://arxiv.org/pdf/2401.06949)  
  *Focus: Robotic assistant for automated chemistry experiments.*
- [TAIS: A Team of AI-made Scientists for Scientific Discovery from Gene Expression Data](https://arxiv.org/pdf/2402.12391)  
  *Focus: AI team for gene expression data analysis.*
- [AI Agents That Matter: Trustworthy Agentic AI for Scientific Discovery](https://arxiv.org/pdf/2407.01502)  
  *Focus: Trustworthy AI agents for scientific discovery.*
- [CRISPR-GPT: An LLM Agent for Automated Design of Gene-Editing Experiments](https://arxiv.org/pdf/2404.18021)  
  *Focus: Automating gene-editing experiment design with LLMs.*
- [BioDiscoveryAgent: An AI Agent for Designing Genetic Perturbation Experiments](https://arxiv.org/pdf/2405.17631)  
  *Focus: AI for designing genetic perturbation experiments.*
- [The Virtual Lab: AI Agents Design New SARS-COV-2 Nanobodies with Experimental Validation.](https://www.biorxiv.org/content/10.1101/2024.11.11.623004v1.full.pdf)
  *Focus: Facilitates AI-human collaboration to design SARS-CoV-2 nanobody binders through interdisciplinary research and task organization.*



#### Challenges and Open Problems

- [SciLitLLM: How to Adapt LLMs for Scientific Literature Understanding](https://arxiv.org/pdf/2408.15545)  
  *Focus: Adapting LLMs for understanding scientific literature.*
- [OpenDevin: An Open Platform for AI Software Developers as Generalist Agents](https://arxiv.org/pdf/2407.16741)  
  *Focus: Open platform for generalist AI software developers.*
- [AI Agents That Matter: Trustworthy Agentic AI for Scientific Discovery](https://arxiv.org/pdf/2407.01502)  
  *Focus: Trustworthy agentic AI for impactful discoveries.*



## Contributing

Feel free to contribute by adding new papers or suggesting improvements! Open a pull request or submit an issue. 😊



## License

[MIT](LICENSE)
