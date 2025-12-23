# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This project is a POC of deploying a flexible LLM-based recommender.
- For a given catalogue, we first use RQ-VAE to learn semantic IDs for each document
- We then fine-tune a small LLM (`3B to 8B` range) to learn the semantic IDs
- We can then use constrained generation to recommend items

This repo should contain:
- Training code for RQ-VAE
- Fine-tuning code for the LLM
- Scripts to run these based on .yaml configurations
- Deployment scripts to deploy the model on a suitable endpoint
- Front-end hosting script with a simple UI to call the model

## Test configuration

Maintain a minimal set of tests that tests only the critical logic of the modules. Aim for readable and essential tests that are easier to maintain rather than exhausting all possibilities.

Before running tests, make sure to `source .venv/bin/activate` before running python scripts.

## Library Usage

Use established implementations when they are available:
- For RQVAE, use `lucidrains/vector-quantize-pytorch`
- For SFT fine-tuning, use `unsloth`
