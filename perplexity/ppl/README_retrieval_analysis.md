# Retrieval Perplexity Analysis Script

This script analyzes log files to extract retrieval information and match it with perplexity scores from a JSON file.

## Usage

```bash
python analyze_retrieval_perplexity_final.py <log_file_path> <ppl_file_path> <output_file_path>
```

### Parameters

- `log_file_path`: Path to the log file containing retrieval information
- `ppl_file_path`: Path to the JSON file containing perplexity scores
- `output_file_path`: Path where the analysis results will be saved

### Example

```bash
python analyze_retrieval_perplexity_final.py top-3/ct-s-3-32/log_0.00_1.00.log traj_abs_format_ppl.json top-3/ct-s-3-32/retrieval_perplexity_analysis.json
```

## Output Format

The script generates a JSON file with the following structure:

```json
{
  "question_id": {
    "summary": {
      "total_steps": 49,
      "status": "unknown",
      "min_perplexity": 6.23826789855957,
      "max_perplexity": 13.528554916381836,
      "mean_perplexity": 9.961965169906616,
      "median_perplexity": 9.834941864013672,
      "std_perplexity": 1.7422075694326096,
      "total_retrievals": 300,
      "valid_retrievals_count": 300
    },
    "detailed_retrievals": [
      {
        "qid": "5a8b3936-43e0-4474-ac15-efaf488265a1",
        "perplexity": 8.809171676635742
      },
      ...
    ]
  }
}
```

## Summary Statistics

For each question, the script provides:

- `total_steps`: Total number of steps actually executed for this specific question (counts "== step: X" occurrences in the question's section)
- `status`: Task completion status ("success" if "finish successfully" found in the question's section, "fail" otherwise)
- `min_perplexity`: Minimum perplexity score among retrieved items
- `max_perplexity`: Maximum perplexity score among retrieved items
- `mean_perplexity`: Average perplexity score
- `median_perplexity`: Median perplexity score
- `std_perplexity`: Standard deviation of perplexity scores
- `total_retrievals`: Total number of retrievals for this question
- `valid_retrievals_count`: Number of retrievals with valid perplexity scores

## Detailed Retrievals

The `detailed_retrievals` array contains each retrieved qid along with its corresponding perplexity score, allowing for detailed analysis of the retrieval patterns.

## Global Statistics

The output JSON includes a `_global_statistics` section that provides comprehensive statistics comparing success and failure cases:

- **Count**: Number of successful and failed questions
- **Steps Statistics**: Min, max, mean, median, and standard deviation of steps for each group
- **Perplexity Statistics**: Min, max, mean, median, and standard deviation of perplexity scores for each group

This allows for easy comparison between successful and failed question executions.

## Requirements

- Python 3.6+
- Standard library modules: json, re, os, statistics, sys, typing, pathlib

## Status Detection

The script automatically detects the success/failure status of each question by:

1. **Default Status**: All questions start with "fail" status
2. **Success Detection**: If "finish successfully" (case-insensitive) appears anywhere in the question's section of the log file, the status is changed to "success"
3. **Section Boundaries**: Each question's section is defined from its initialization line to the next question's initialization line

## Step Counting

The script counts the actual number of steps executed for each question by:

1. **Step Detection**: Looks for lines containing "== step: X" in each question's section
2. **Maximum Step**: Records the highest step number found for each question
3. **Individual Counting**: Each question's step count is calculated independently, not based on the global maximum

## Notes

- The script automatically associates retrievals with questions based on line numbers in the log file
- Only retrievals with valid perplexity scores in the provided JSON file are included in the analysis
- The script handles missing perplexity data gracefully by setting statistics to null when no valid retrievals are found
- Status detection is based on the presence of the word "fail" in the question's log section
