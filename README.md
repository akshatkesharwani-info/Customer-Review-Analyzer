# Customer Review Analyzer

Turns thousands of unread product reviews into a structured summary of complaint themes and
feature requests -- so a product team doesn't have to read every single one.

## Problem
A retailer has 23,000+ product reviews sitting unread. The signal about sizing issues, fabric
quality, and feature requests is buried in there, but nobody has time to read it all manually.

## What It Does
- Scores sentiment on every review using a local transformer model (no API cost)
- Runs topic modeling (LDA) specifically on 1-2 star reviews to isolate real pain points
- Sends a sample of negative reviews to a free-tier LLM to synthesize a plain-English summary
  of the top complaints and feature requests
- Breaks sentiment down by department and by customer age bracket

## Real Results (real dataset, 23,486 women's clothing reviews)
- Topic modeling surfaced clear, distinct complaint clusters: **fit/sizing issues**,
  **fabric/material quality**, **wrong expectations vs. photos**
- LLM-synthesized recommendations: audit the size chart and cut patterns (especially sleeves/
  shoulders/length), shift material sourcing to reduce chemical odor complaints, and tighten QC
  on seams, zippers, and missing hardware like elastic waistbands

## Tech Stack
Python, HuggingFace Transformers, Scikit-learn (TF-IDF/LDA), Cerebras/Groq free tier

## How to Run
Open in Google Colab, run all cells. Note: sentiment scoring runs a transformer over every row,
so for large datasets consider sampling down (e.g. 800 reviews) to keep runtime reasonable.
