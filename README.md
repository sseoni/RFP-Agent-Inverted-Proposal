# RFP-Agent-Inverted-Proposal

## Overview

This project aims to develop a system that generates RFPs (Requests for Proposals) or Inverted Proposals by retrieving relevant information from a database based on the topic provided by the user. By combining data retrieval and automated generation, the system enables users to quickly produce high-quality RFPs with minimal input.

## Problem

According to research by [Loopio](https://www.marketingprofs.com/charts/2020/42512/rfp-benchmarks-how-much-time-and-staff-firms-devote-to-proposals?utm_source=chatgpt.com), companies spend an average of 23.8 hours writing a single Request for Proposal (RFP), with larger firms taking up to 35.2 hours. This process typically involves multiple team members and can extend over several weeks, especially when accounting for data collection, supplier selection, and proposal evaluation stages. Such extensive time commitments can strain resources and prolong project timelines.

## Solution

Our solution focuses on database retrieval and automated RFP generation, consisting of the following key steps:

1. **Database Retrieval**
    - Relevant projects, existing RFPs, and similar data are retrieved from the database based on the user’s provided topic.
    - Retrieval-Augmented Generation (RAG) technology ensures relevance and prioritization of the retrieved data.
2. **Automated RFP Generation**
    - Using the retrieved data, the system automatically generates an RFP tailored to the user’s topic.
    - The output aligns with the user’s requirements and ecosystem needs.
3. **On-Chain Data Integration**
    - The generated RFP is integrated with on-chain data to ensure transparency and reliability.
    - AI-PGF forum contract data is referenced to validate consistency.
4. **Result Storage and Utilization**
    - The generated RFP is stored in the database for future reference or submission.
    - Users can reuse or modify the stored RFPs for subsequent projects.
