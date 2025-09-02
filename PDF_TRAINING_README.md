# PDF Bot Trainer

Train your Arvocap chatbot with PDF documents containing investment reports, fund performance data, regulatory documents, and any other financial content.

## Quick Start

### Option 1: Double-click the batch file (Windows)
```
Double-click: train_pdfs.bat
```

### Option 2: Run Python script directly
```bash
# Interactive mode (recommended for first time)
python pdf_trainer.py

# Automatic mode
python pdf_trainer.py --auto

# Clear existing data and retrain
python pdf_trainer.py --auto --clear
```

## How It Works

1. **Add PDFs**: Place your PDF files in the `data/pdfs/` directory
2. **Run Trainer**: Execute the training program
3. **Processing**: The system will:
   - Extract text from all PDFs
   - Split content into manageable chunks
   - Create vector embeddings
   - Update the chatbot's knowledge base
4. **Test**: Your chatbot can now answer questions about the PDF content

## PDF Directory Structure

```
python_training/
├── data/
│   └── pdfs/              # Place your PDF files here
│       ├── fund_report_2024.pdf
│       ├── investment_strategy.pdf
│       └── regulatory_filing.pdf
├── pdf_trainer.py         # Main training script
└── train_pdfs.bat        # Windows batch file
```

## Supported PDF Types

- Investment fund reports
- Financial statements
- Regulatory filings
- Research reports
- Company documentation
- Market analysis
- Any text-based PDF document

## Training Options

### Interactive Mode (Default)
```bash
python pdf_trainer.py
```
- Shows found PDF files
- Asks for confirmation
- Provides detailed progress updates

### Automatic Mode
```bash
python pdf_trainer.py --auto
```
- Processes all PDFs automatically
- No user interaction required
- Good for automation/scripts

### Custom PDF Directory
```bash
python pdf_trainer.py --pdf-dir "C:\My Documents\PDFs"
```
- Use a different directory for PDF files
- Useful if you have PDFs in multiple locations

### Clear Existing Data
```bash
python pdf_trainer.py --clear
```
- Removes all existing training data
- Starts fresh with only new PDFs
- Use when you want to completely retrain

## Example Training Session

```
📚 PDF Bot Trainer - Interactive Mode
============================================================
📁 PDF Directory: C:\...\data\pdfs

📄 Found 3 PDF files:
   1. arvocap_fund_report_2024.pdf (2.3 MB)
   2. market_analysis_q3.pdf (1.8 MB)
   3. regulatory_update.pdf (0.9 MB)

🤖 This will train the chatbot on these PDF documents.
⚠️  Current knowledge base has 15 documents.
Clear existing data? (y/N): n

Proceed with training? (Y/n): y

🔄 Starting training...
🔄 Processing PDF files...
Processing: arvocap_fund_report_2024.pdf
  ✅ Processed arvocap_fund_report_2024.pdf: 45 chunks
Processing: market_analysis_q3.pdf
  ✅ Processed market_analysis_q3.pdf: 32 chunks
Processing: regulatory_update.pdf
  ✅ Processed regulatory_update.pdf: 18 chunks

🔍 Updating knowledge base...
✅ Added 95 chunks to knowledge base
📊 Total documents in knowledge base: 110

==================================================
🎉 PDF Training Completed Successfully!
📊 Training Summary:
   - PDFs processed: 3
   - Total pages: 87
   - Total chunks: 95
   - Knowledge base size: 110

🚀 Your chatbot is now trained on the PDF content!
💬 Test it by running: python api_server.py
```

## Testing Your Trained Bot

After training, test the chatbot:

1. **Start the API server**:
   ```bash
   python api_server.py
   ```

2. **Test with questions about your PDFs**:
   - "What was the fund performance in 2024?"
   - "Tell me about the investment strategy"
   - "What are the key regulatory changes?"

## Troubleshooting

### No PDF files found
- Make sure PDF files are in the `data/pdfs/` directory
- Check file permissions
- Ensure files have `.pdf` extension

### Import errors
- Activate the virtual environment: `env\Scripts\activate`
- Install dependencies: `pip install -r requirements.txt`

### Processing errors
- Check PDF file integrity
- Ensure PDFs contain extractable text (not just images)
- Try with a smaller PDF first

### Memory issues
- Process PDFs one at a time for large files
- Use `--clear` to start fresh if needed

## Advanced Usage

### Multiple Training Sessions
You can run training multiple times to add more PDFs:
1. Add new PDFs to the directory
2. Run trainer again (don't use `--clear`)
3. New content will be added to existing knowledge

### Batch Processing
For automated training:
```bash
# Copy PDFs and train automatically
copy "\\server\reports\*.pdf" "data\pdfs\"
python pdf_trainer.py --auto
```

### Integration with Web Interface
After training, your PDFs content will be available through:
- The web chatbot interface
- API endpoints
- Direct chat functionality

## File Outputs

The trainer creates these files:
- `data/processed_pdfs.json` - Processed PDF content
- `vector_db/` - Vector database with embeddings
- Training logs in the console

Your chatbot is now ready to answer questions about your PDF content!
