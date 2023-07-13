#!/bin/bash

source_directories=(
    # "source_documents/MITCourse/18-657-mathematics-of-machine-learning-fall-2015"
    # "source_documents/MITCourse/18-06sc-linear-algebra-fall-2011"
    # "source_documents/MITCourse/15-450-analytics-of-finance-fall-2010"
    # "source_documents/MITCourse/18-01-calculus-i-single-variable-calculus-fall-2020"
    # "source_documents/MITCourse/18-02sc-multivariable-calculus-fall-2010"
    # "source_documents/MITCourse/18-03sc-differential-equations-fall-2011"
    # "source_documents/MITCourse/2-087-engineering-math-differential-equations-and-linear-algebra-fall-2014"
    # "source_documents/MITCourse/6-871-knowledge-based-applications-systems-spring-2005"
    # "source_documents/MITCourse/6-036-introduction-to-machine-learning-fall-2020"
    # "source_documents/MITCourse/9-520-statistical-learning-theory-and-applications-spring-2003"
    # "source_documents/MITCourse/18-650-statistics-for-applications-fall-2016"
    # "source_documents/MITCourse/6-801-machine-vision-fall-2020"
    # "source_documents/MITCourse/15-s08-fintech-shaping-the-financial-world-spring-2020"
    # "source_documents/MITCourse/18-s096-topics-in-mathematics-with-applications-in-finance-fall-2013"
    # "source_documents/MITCourse/2-035-special-topics-in-mathematics-with-applications-linear-algebra-and-the-calculus-of-variations-spring-2007"
    # "source_documents/MITCourse/6-041sc-probabilistic-systems-analysis-and-applied-probability-fall-2013"
    # "source_documents/MITCourse/6-262-discrete-stochastic-processes-spring-2011"
    # "source_documents/MITCourse/6-034-artificial-intelligence-fall-2010"
    # "source_documents/MITCourse/18-065-matrix-methods-in-data-analysis-signal-processing-and-machine-learning-spring-2018"
    # "source_documents/MITCourse/res-ll-005-mathematics-of-big-data-and-machine-learning-january-iap-2020"
    # "source_documents/MITCourse/15-071-the-analytics-edge-spring-2017"
    # "source_documents/MITCourse/6-253-convex-analysis-and-optimization-spring-2012"
    # "source_documents/MITCourse/15-093j-optimization-methods-fall-2009"
    # "source_documents/MITCourse/6-863j-natural-language-and-the-computer-representation-of-knowledge-spring-2003"
    # "source_documents/MITCourse/9-98-language-and-mind-january-iap-2003"
    # "source_documents/MITCourse/6-864-advanced-natural-language-processing-fall-2005"
    # "source_documents/MITCourse/9-913-pattern-recognition-for-machine-vision-fall-2004"
    "source_documents/docs"
)
exception_directory="content"

for source_directory in "${source_directories[@]}"; do
    for entry in "$source_directory"/*; do
        if [[ -d "$entry" ]]; then
            # Skip the exception directory
            if [[ "${entry##*/}" == "$exception_directory" ]]; then
                continue
            fi

            # Remove non-exception directories recursively
            echo "Removing directory: $entry"
            sudo rm -rf "$entry"
        elif [[ -f "$entry" ]]; then
            # Remove files
            sudo rm "$entry"
        fi
    done
done
