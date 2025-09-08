import pandas as pd
import time
import logging
from pathlib import Path
import argparse
from predict import EmailSpamPredictor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BatchEmailProcessor:
    """
    A class for processing large batches of emails efficiently.
    """

    def __init__(self, model_path='./models', chunk_size=1000):
        """
        Initialize the batch processor.

        Args:
            model_path (str): Path to trained models
            chunk_size (int): Number of emails to process in each chunk
        """
        self.predictor = EmailSpamPredictor(model_path)
        self.chunk_size = chunk_size
        self.results_summary = {
            'total_processed': 0,
            'spam_detected': 0,
            'ham_detected': 0,
            'errors': 0,
            'processing_time': 0
        }

    def process_csv_file(self, input_file, output_file=None, progress_callback=None):
        """
        Process emails from a CSV file in chunks.

        Args:
            input_file (str): Path to input CSV file
            output_file (str, optional): Path to output CSV file
            progress_callback (callable, optional): Function to call with progress updates

        Returns:
            pd.DataFrame: Processed results
        """
        logger.info(f"Starting batch processing of {input_file}")
        start_time = time.time()

        # Generate output filename if not provided
        if output_file is None:
            input_path = Path(input_file)
            output_file = input_path.parent / f"{input_path.stem}_predictions{input_path.suffix}"

        # Get file size for progress tracking
        try:
            total_rows = sum(1 for _ in open(input_file)) - 1  # Subtract header
            logger.info(f"Processing {total_rows:,} emails in chunks of {self.chunk_size}")
        except:
            total_rows = None
            logger.warning("Could not determine file size for progress tracking")

        processed_chunks = []
        chunk_count = 0

        # Process file in chunks
        for chunk in pd.read_csv(input_file, chunksize=self.chunk_size):
            chunk_count += 1
            chunk_start_time = time.time()

            logger.info(f"Processing chunk {chunk_count} ({len(chunk)} emails)")

            try:
                # Process chunk
                processed_chunk = self.process_chunk(chunk)
                processed_chunks.append(processed_chunk)

                # Update summary
                self.update_summary(processed_chunk)

                # Calculate progress
                if total_rows and progress_callback:
                    progress = min((chunk_count * self.chunk_size) / total_rows * 100, 100)
                    progress_callback(progress, chunk_count, len(chunk))

                chunk_time = time.time() - chunk_start_time
                logger.info(f"Chunk {chunk_count} completed in {chunk_time:.2f}s")

            except Exception as e:
                logger.error(f"Error processing chunk {chunk_count}: {str(e)}")
                self.results_summary['errors'] += len(chunk)

        # Combine all chunks
        if processed_chunks:
            final_results = pd.concat(processed_chunks, ignore_index=True)

            # Save results
            final_results.to_csv(output_file, index=False)
            logger.info(f"Results saved to {output_file}")

            # Calculate final statistics
            total_time = time.time() - start_time
            self.results_summary['processing_time'] = total_time
            self.print_summary()

            return final_results

        else:
            logger.error("No data was processed successfully")
            return pd.DataFrame()

    def process_chunk(self, chunk):
        """
        Process a single chunk of emails.

        Args:
            chunk (pd.DataFrame): Chunk of emails to process

        Returns:
            pd.DataFrame: Processed chunk with predictions
        """
        predictions = []

        for idx, row in chunk.iterrows():
            try:
                # Extract email components
                email_text = str(row.get('body', ''))
                sender = str(row.get('sender', ''))
                subject = str(row.get('subject', ''))
                urls = int(row.get('urls', 0))

                # Make prediction
                result = self.predictor.predict_email(email_text, sender, subject, urls)

                # Store prediction
                predictions.append({
                    'prediction': result['prediction'],
                    'confidence': result['confidence'],
                    'predicted_label': result['label'],
                    'model_used': result.get('model_used', 'Unknown')
                })

            except Exception as e:
                logger.warning(f"Error processing email at index {idx}: {str(e)}")
                predictions.append({
                    'prediction': 'Error',
                    'confidence': 0.0,
                    'predicted_label': 'Error',
                    'model_used': 'Error'
                })

        # Add predictions to chunk
        for key in ['prediction', 'confidence', 'predicted_label', 'model_used']:
            chunk[f'predicted_{key}'] = [pred[key] for pred in predictions]

        return chunk

    def update_summary(self, processed_chunk):
        """Update processing summary statistics."""
        self.results_summary['total_processed'] += len(processed_chunk)

        spam_count = sum(1 for pred in processed_chunk['predicted_prediction'] 
                        if pred == 'Spam')
        ham_count = sum(1 for pred in processed_chunk['predicted_prediction'] 
                       if pred == 'Not Spam')
        error_count = sum(1 for pred in processed_chunk['predicted_prediction'] 
                         if pred == 'Error')

        self.results_summary['spam_detected'] += spam_count
        self.results_summary['ham_detected'] += ham_count
        self.results_summary['errors'] += error_count

    def print_summary(self):
        """Print processing summary."""
        summary = self.results_summary

        print("\n" + "="*60)
        print("BATCH PROCESSING SUMMARY")
        print("="*60)
        print(f"Total emails processed: {summary['total_processed']:,}")
        print(f"Spam detected: {summary['spam_detected']:,} ({summary['spam_detected']/summary['total_processed']*100:.1f}%)")
        print(f"Ham detected: {summary['ham_detected']:,} ({summary['ham_detected']/summary['total_processed']*100:.1f}%)")
        print(f"Errors: {summary['errors']:,} ({summary['errors']/summary['total_processed']*100:.1f}%)")
        print(f"Processing time: {summary['processing_time']:.2f} seconds")
        print(f"Processing rate: {summary['total_processed']/summary['processing_time']:.1f} emails/second")
        print("="*60)

    def process_directory(self, input_dir, output_dir=None, file_pattern="*.csv"):
        """
        Process all CSV files in a directory.

        Args:
            input_dir (str): Input directory path
            output_dir (str, optional): Output directory path
            file_pattern (str): File pattern to match

        Returns:
            list: List of processed files
        """
        input_path = Path(input_dir)

        if output_dir is None:
            output_path = input_path / "predictions"
        else:
            output_path = Path(output_dir)

        # Create output directory
        output_path.mkdir(exist_ok=True)

        # Find all matching files
        input_files = list(input_path.glob(file_pattern))

        if not input_files:
            logger.warning(f"No files found matching pattern {file_pattern} in {input_dir}")
            return []

        logger.info(f"Found {len(input_files)} files to process")

        processed_files = []

        for input_file in input_files:
            logger.info(f"Processing file: {input_file.name}")

            output_file = output_path / f"{input_file.stem}_predictions{input_file.suffix}"

            try:
                result_df = self.process_csv_file(str(input_file), str(output_file))
                processed_files.append({
                    'input_file': str(input_file),
                    'output_file': str(output_file),
                    'status': 'success',
                    'rows_processed': len(result_df)
                })
            except Exception as e:
                logger.error(f"Failed to process {input_file.name}: {str(e)}")
                processed_files.append({
                    'input_file': str(input_file),
                    'output_file': None,
                    'status': 'error',
                    'error': str(e)
                })

        return processed_files

def progress_callback(progress, chunk_num, chunk_size):
    """Simple progress callback function."""
    print(f"Progress: {progress:.1f}% (Chunk {chunk_num}, {chunk_size} emails)")

def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description='Email Spam Classifier - Batch Processing')
    parser.add_argument('input', help='Input CSV file or directory')
    parser.add_argument('-o', '--output', help='Output file or directory')
    parser.add_argument('-c', '--chunk-size', type=int, default=1000, 
                       help='Chunk size for processing (default: 1000)')
    parser.add_argument('-m', '--model-path', default='./models',
                       help='Path to trained models (default: ./models)')
    parser.add_argument('--directory', action='store_true',
                       help='Process all CSV files in a directory')

    args = parser.parse_args()

    try:
        # Initialize processor
        processor = BatchEmailProcessor(args.model_path, args.chunk_size)

        if args.directory:
            # Process directory
            processed_files = processor.process_directory(args.input, args.output)

            print(f"\nProcessed {len(processed_files)} files:")
            for file_info in processed_files:
                status = file_info['status']
                if status == 'success':
                    print(f"  ✓ {file_info['input_file']} -> {file_info['output_file']} ({file_info['rows_processed']} rows)")
                else:
                    print(f"  ✗ {file_info['input_file']} - Error: {file_info.get('error', 'Unknown')}")
        else:
            # Process single file
            result_df = processor.process_csv_file(args.input, args.output, progress_callback)
            print(f"\nProcessing completed. {len(result_df)} emails processed.")

    except Exception as e:
        logger.error(f"Batch processing failed: {str(e)}")
        return 1

    return 0

if __name__ == "__main__":
    exit(main())

# Example usage:
# 
# Process single file:
# python batch_processing.py emails.csv -o predictions.csv
# 
# Process directory:
# python batch_processing.py ./data --directory -o ./predictions
# 
# Custom chunk size:
# python batch_processing.py large_dataset.csv -c 5000
