# Feature: {feature_description}

## Overview
Auto-generated workflow for implementing {feature_description}.

## Prerequisites
- [ ] Codebase analyzed
- [ ] Dependencies identified

## Phase 1: Analysis
- [ ] Step 1.1: Run query_codebase(questions=[...])
  - Questions to ask:
    * "How is {related_feature} currently implemented?"
    * "What are the existing patterns for {pattern_type}?"
- [ ] Step 1.2: Review analysis results
- [ ] Step 1.3: Run find_code_by_intent(intent="{intent}")

## Phase 2: Planning (Claude's Job)
- [ ] Step 2.1: Create detailed specification using analysis
  - Use facts from Phase 1
  - Follow existing patterns
  - Define clear acceptance criteria
- [ ] Step 2.2: Run validate_against_codebase(spec=...)
- [ ] Step 2.3: Address validation issues

## Phase 3: Implementation
- [ ] Step 3.1: Install dependencies
  - `npm install {dependencies}` or `pip install {dependencies}`
- [ ] Step 3.2: Create new files
  - {file_list}
- [ ] Step 3.3: Modify existing files
  - {file_list}
- [ ] Step 3.4: Update tests

## Phase 4: Validation
- [ ] Step 4.1: Run tests
- [ ] Step 4.2: Run check_consistency(focus="all")
- [ ] Step 4.3: Address inconsistencies

## Phase 5: Documentation
- [ ] Step 5.1: Update README
- [ ] Step 5.2: Add API documentation
- [ ] Step 5.3: Update changelog
