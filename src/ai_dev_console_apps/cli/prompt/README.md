# ai-prompt

Run `ai-prompt --help` to see the available options and environment variables to set.

## Usage

```bash
# anthropic
echo "Hi" | ai-prompt --vendor anthropic --model claude-3-haiku-20240307

echo "Hi" | ai-prompt --vendor anthropic --model claude-3-haiku-20240307 --temperature 0.8 --max-tokens 2000

# Basic Bedrock prompt
echo "Hi" | ai-prompt --vendor aws --model anthropic.claude-3-haiku-20240307-v1:0

# With custom temperature
echo "Hi" | ai-prompt --vendor aws --model anthropic.claude-3-haiku-20240307-v1:0 --temperature 0.7

# With max tokens
echo "Hi" | ai-prompt --vendor aws --model anthropic.claude-3-haiku-20240307-v1:0 --max-tokens 1500

# Combining parameters
echo "Hi" | ai-prompt --vendor aws --model anthropic.claude-3-haiku-20240307-v1:0 --temperature 0.8 --max-tokens 2000
```
