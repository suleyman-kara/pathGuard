## Summary of the Competition Progress and Next Steps [00:00 – 04:40]

The first phase of the competition, including the initial project result reports, data sharing, and evaluations, has concluded. The second phase involves submitting more general reports with fewer and broader questions, encouraging creativity from participants. These reports will again be reviewed by judges, who consider not only the technical content but also report quality—such as consistent writing style, appropriate use of headings, visual aids, and proper referencing.

Around 200-250 teams participated initially, from which approximately 40 teams are expected to advance to the final stage. The final phase will emphasize model performance on test datasets structurally identical to training datasets but without labels. Teams must prepare four separate models, each tailored to one of four distinct data panels or subsets, producing separate scores which will then be combined for final ranking.

The final evaluation will include a small weight (~10%) for presentation quality. The exact timing and duration for the final testing phase are yet to be determined, but it will likely be short, requiring models to be fully prepared beforehand.

## Report Preparation and Evaluation Criteria [01:08 – 02:55]

- The **second phase reports** should focus on **data analysis** based on the provided datasets.
- Reports should exhibit originality beyond AI-generated content to ensure higher scores, as judges are increasingly experienced with generic AI outputs.
- Report formatting, including consistent writing styles, clear section headings, visual supports, and correctly formatted references/bibliographies, critically impacts scoring.
- Teams have a page limit (approximately 10 pages) to concisely present their findings and analyses.
- Including code is optional and depends on the team’s discretion.

## Data and Modeling Details [04:59 – 07:54]

- Four separate datasets (“panels”) are provided, requiring four **distinct models**, not a single unified model.
- Evaluation metrics such as precision, recall, F1, and ROC will focus specifically on the **pathogenic** class (label 1), not benign (label 0) separately.
- Training data distribution: approximately 80% pathogenic and 20% benign.
- Test data distribution is approximately reversed: 20% pathogenic and 80% benign, a realistic and challenging setup.
- Model scores for each of the four panels will be combined for the final ranking.
- Use of **synthetic data is permitted**, but the competition primarily assesses performance on provided datasets.

## Data Characteristics and Handling Missing Values [11:18 – 18:50]

- Training data contains deliberate **missing values**; participants are expected to address these through preprocessing or other strategies.
- Zero values and missing values should be treated differently during data handling.
- Columns in the dataset provide varying types of information; some include protein sequences and epigenetic context, though detailed contextual sequence data (neighboring amino acids or nucleotides) has been removed to maintain challenge difficulty.
- Participants may integrate additional external datasets but are encouraged to focus primarily on the provided data.
- Labels are binary: 1 for pathogenic variants and 0 for benign, consistent with dataset documentation.
- Some variants may appear mislabeled or without changes (e.g., nucleotide-to-same-nucleotide), which should be excluded or handled cautiously.

## Model Expectations and Performance Evaluation [29:49 – 31:55]

- Each panel requires its **own model** rather than a single model handling all panels.
- Models should output direct pathogenicity predictions (binary classification) rather than probability scores.
- The final stage test will use unlabeled data; after generating predictions, performance will be assessed using the pathogenic class-based F1 score.
- The competition favors realistic, challenging setups reflecting practical genetic variant classification, supporting the choice of unbalanced distributions.
- Explainability and feature importance comments are expected in reports but are not mandatory in final evaluations, which focus on predictive performance.

## Frequently Asked Questions and Clarifications [20:00 – 35:00]

|Question/Topic|Clarification/Answer|
|---|---|
|Can external synthetic data be used?|Yes, but the main assessment is based on provided datasets.|
|Are final evaluation metrics only F1?|F1 focused on pathogenic variants, with metrics like ROC also usable.|
|Should amino acid and nucleotide context be included?|Such data is no longer shared to maintain challenge difficulty.|
|How to treat missing or zero values?|Treat missing and zero differently; zeros are not missing data.|
|Are code submissions mandatory?|No, only reports are required; adding code is optional.|
|Will hidden test sets be provided?|Yes, test sets with unknown labels will be used in the final evaluation.|

## Timeline and Practical Information [00:32 – 03:30, 30:39 – 31:00]

- Phase 2 reports are to be submitted soon, judged by experts.
- Approximately 40 teams will proceed to the finals.
- Final test set and evaluation details, including infrastructure specs, will be shared closer to final stage.
- Competition is organized to ensure reproducibility; final codes may be re-run later.
- The organizers encourage participants to check the latest updated competition rules posted on the Teknofest website.

## Key Takeaways and Recommendations for Participants

- **Focus on clarity and originality** in reports; avoid generic AI-generated text.
- Develop **four separate models**, one for each panel, and optimize for **pathogenic variant detection** with a focus on F1 score.
- Handle missing and zero data thoughtfully—not all zeroes mean missing.
- Preprocess data as preferred but ensure consistency between training and test dataset formats.
- Use extra external data cautiously; the competition primarily values performance on given datasets.
- Prepare models and pipelines to quickly generate predictions during a final timed test phase.
- Presentation skills contribute to final scoring (approximate 10%) so plan a clear, concise final presentation.
- Review updated competition rules and datasets regularly to avoid misinterpretations.

This phase of the competition aims not only to benchmark predictive models but also to deepen participant understanding of genetic variant data characteristics and the challenges in applying AI to this domain. The competition places strong emphasis on real-world applicability and data-driven creativity.
