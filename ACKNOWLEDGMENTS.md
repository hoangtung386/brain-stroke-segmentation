# Acknowledgments

This project would not have been possible without the invaluable assistance and contributions from various tools, technologies, and communities.

> [!IMPORTANT]
> ## PRIMARY AUTHOR & DEVELOPER
> 
> **All core implementation, architectural design, and scientific contributions were created by:**
> ### Le Vu Hoang Tung
> 
> **What the primary author developed:**
> - **Complete LCNN architecture** - Designed and implemented from scratch
> - **SEAN (Symmetry Enhanced Attention Network)** - Full implementation including alignment network
> - **Entire training pipeline** - Data loading, loss functions, optimization strategy
> - **Model architecture integration** - Combined local-global pathways, ResNeXt backbone
> - **Problem solving** - Resolved gradient explosion, NaN issues, dimension mismatches
> - **Debugging complex errors** - Fixed training instabilities on lab workstation
> - **Experiment design** - Hyperparameter tuning, validation strategy

---

## AI Assistants (Support Role Only)

The following AI tools provided **AUXILIARY SUPPORT** during development:

### Google Gemini 3.0 Pro
- Debugging support when errors occurred during training
- Analysis of error messages and stack traces
- Suggestions for gradient clipping and loss scaling
- Discussion of potential causes for training instabilities

### Anthropic Claude Sonnet 4.5
- Code review and analysis assistance
- Help writing evaluation and visualization functions
- Explanations of tensor operations when debugging
- Suggestions for error handling patterns

---

> [!WARNING]
> ## CRITICAL CLARIFICATION 
> 
> ### What AI Did NOT Do:
> - Did NOT design the architecture
> - Did NOT write the core model code
> - Did NOT implement LCNN or SEAN
> - Did NOT develop the training strategy
> - Did NOT make scientific or architectural decisions
> 
> ### What AI DID Do:
> - Helped debug errors during training runs
> - Assisted with writing evaluation/visualization utilities
> - Provided explanations when analyzing bugs
> - Acted as a "rubber duck" for problem-solving discussions
> 

---

## Technologies & Frameworks

This project is built upon excellent open-source frameworks and libraries:

- **PyTorch**: The foundation of our deep learning implementation
- **MONAI**: Medical imaging-specific utilities and loss functions
- **torchvision**: ResNeXt backbone and image transformations
- **Weights & Biases**: Experiment tracking and visualization

---

## Research Foundations

This work is inspired by and builds upon research in:
- Symmetry-enhanced attention mechanisms for medical imaging
- Local-global combined neural networks for segmentation
- Brain stroke lesion detection methodologies

---

## Community

Thanks to the broader deep learning and medical imaging communities for:
- Open-source tools and libraries
- Research papers and methodologies
- Discussion forums and knowledge sharing platforms

---

## License

This project is licensed under the MIT License - see the [LICENSE.txt](LICENSE.txt) file for details.

The acknowledgment of AI assistance does not transfer any intellectual property rights to these tools or their creators. All code in this repository remains under the project's MIT License.

---

*If you use this project in your research or work, please consider citing both this repository and acknowledging the collaborative nature of modern software development.*
