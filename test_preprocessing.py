"""
Test suite for preprocessing functionality.

Tests the preprocessing module to ensure consistent and correct state preprocessing
for both CPU and GPU implementations.
"""
import unittest
import numpy as np
import torch
import sys
import os

# Add current directory to path to import local modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from preprocessing import (
        preprocess_state, BatchPreprocessor, get_state_shape, 
        validate_state_format, create_preprocessor
    )
    PREPROCESSING_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import preprocessing module: {e}")
    PREPROCESSING_AVAILABLE = False

try:
    from simple_tetris_env import SimpleTetrisEnv, TETROMINO_SHAPES
    ENV_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import SimpleTetrisEnv: {e}")
    ENV_AVAILABLE = False


class TestPreprocessing(unittest.TestCase):
    """Test cases for preprocessing functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        if not PREPROCESSING_AVAILABLE:
            self.skipTest("Preprocessing module not available")
    
    def test_basic_preprocessing(self):
        """Test basic preprocessing without piece information."""
        # Create simple grid state
        grid = np.random.randint(0, 2, (14, 7))
        
        # Test basic preprocessing
        result = preprocess_state(grid, include_piece_info=False)
        
        # Check output format
        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(len(result.shape), 3)  # [C, H, W]
        self.assertEqual(result.shape, (1, 14, 7))  # 1 channel for basic
    
    def test_enhanced_preprocessing(self):
        """Test enhanced preprocessing with piece information."""
        # Create state dictionary
        state = {
            'grid': np.random.randint(0, 2, (14, 7)),
            'current_piece': 1,
            'piece_x': 3,
            'piece_y': 2,
            'piece_rotation': 0,
            'next_piece': 2
        }
        
        result = preprocess_state(state, include_piece_info=True)
        
        # Check output format
        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(len(result.shape), 3)  # [C, H, W]
        self.assertEqual(result.shape, (4, 14, 7))  # 4 channels for enhanced
    
    def test_binary_conversion(self):
        """Test binary state conversion."""
        # Create grid with various values
        grid = np.array([[0, 1, 2, 3], [4, 5, 0, 1]])
        
        # Test without binary conversion
        result_normal = preprocess_state(grid, binary=False, include_piece_info=False)
        
        # Test with binary conversion
        result_binary = preprocess_state(grid, binary=True, include_piece_info=False)
        
        # Binary version should only have 0s and 1s
        unique_values = torch.unique(result_binary)
        self.assertTrue(torch.all(torch.isin(unique_values, torch.tensor([0.0, 1.0]))))
        
        # Non-binary should preserve original values
        self.assertFalse(torch.equal(result_normal, result_binary))
    
    def test_device_placement(self):
        """Test that tensors are placed on correct device."""
        grid = np.random.rand(14, 7)
        
        # Test CPU placement
        result_cpu = preprocess_state(grid, device="cpu", include_piece_info=False)
        self.assertEqual(result_cpu.device.type, "cpu")
        
        # Test GPU placement if available
        if torch.cuda.is_available():
            result_gpu = preprocess_state(grid, device="cuda", include_piece_info=False)
            self.assertEqual(result_gpu.device.type, "cuda")
    
    def test_dtype_conversion(self):
        """Test data type conversion."""
        grid = np.random.rand(14, 7)
        
        # Test float32
        result_float32 = preprocess_state(grid, dtype=torch.float32, include_piece_info=False)
        self.assertEqual(result_float32.dtype, torch.float32)
        
        # Test float16
        result_float16 = preprocess_state(grid, dtype=torch.float16, include_piece_info=False)
        self.assertEqual(result_float16.dtype, torch.float16)
    
    def test_channel_consistency(self):
        """Test preprocessing produces consistent channels."""
        if not ENV_AVAILABLE:
            self.skipTest("Environment not available for state generation")
        
        env = SimpleTetrisEnv()
        state = env.reset()
        
        # Test with different settings
        basic = preprocess_state(state, include_piece_info=False)
        enhanced = preprocess_state(state, include_piece_info=True)
        
        self.assertEqual(basic.shape[0], 1)  # 1 channel for basic
        self.assertEqual(enhanced.shape[0], 4)  # 4 channels for enhanced
        
        # Both should have same spatial dimensions
        self.assertEqual(basic.shape[1:], enhanced.shape[1:])
    
    def test_none_state_handling(self):
        """Test handling of None states (terminal states)."""
        result = preprocess_state(None, include_piece_info=False)
        self.assertIsNone(result)
    
    def test_tensor_input_handling(self):
        """Test that tensor inputs are handled correctly."""
        # Create tensor input
        tensor_input = torch.rand(14, 7)
        
        result = preprocess_state(tensor_input, include_piece_info=False)
        
        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.shape, (1, 14, 7))


class TestBatchPreprocessor(unittest.TestCase):
    """Test cases for batch preprocessing."""
    
    def setUp(self):
        """Set up test fixtures."""
        if not PREPROCESSING_AVAILABLE:
            self.skipTest("Preprocessing module not available")
    
    def test_batch_preprocessor_initialization(self):
        """Test batch preprocessor initialization."""
        preprocessor = BatchPreprocessor(
            device="cpu",
            binary=True,
            include_piece_info=False
        )
        
        self.assertEqual(preprocessor.device, "cpu")
        self.assertTrue(preprocessor.binary)
        self.assertFalse(preprocessor.include_piece_info)
    
    def test_batch_preprocessing(self):
        """Test batch preprocessing functionality."""
        preprocessor = BatchPreprocessor(device="cpu", include_piece_info=False)
        
        # Create batch of states
        states = [
            np.random.rand(14, 7),
            np.random.rand(14, 7),
            np.random.rand(14, 7)
        ]
        
        result = preprocessor.preprocess_batch(states)
        
        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.shape, (3, 1, 14, 7))  # [B, C, H, W]
    
    def test_batch_with_none_states(self):
        """Test batch preprocessing with None states."""
        preprocessor = BatchPreprocessor(device="cpu", include_piece_info=False)
        
        # Create batch with None states
        states = [
            np.random.rand(14, 7),
            None,  # Terminal state
            np.random.rand(14, 7)
        ]
        
        result = preprocessor.preprocess_batch(states)
        
        # Should filter out None states
        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.shape, (2, 1, 14, 7))
    
    def test_empty_batch_handling(self):
        """Test handling of empty or all-None batches."""
        preprocessor = BatchPreprocessor(device="cpu", include_piece_info=False)
        
        # All None states
        states = [None, None, None]
        result = preprocessor.preprocess_batch(states)
        self.assertIsNone(result)
        
        # Empty batch
        states = []
        result = preprocessor.preprocess_batch(states)
        self.assertIsNone(result)
    
    def test_mixed_input_types(self):
        """Test batch preprocessing with mixed input types."""
        preprocessor = BatchPreprocessor(device="cpu", include_piece_info=False)
        
        # Mix of numpy arrays and tensors
        states = [
            np.random.rand(14, 7),
            torch.rand(1, 14, 7),  # Already preprocessed tensor
            np.random.rand(14, 7)
        ]
        
        result = preprocessor.preprocess_batch(states)
        
        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.shape, (3, 1, 14, 7))
    
    def test_gpu_batch_preprocessing(self):
        """Test GPU batch preprocessing if available."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        
        preprocessor = BatchPreprocessor(device="cuda", include_piece_info=False)
        
        states = [
            np.random.rand(14, 7),
            np.random.rand(14, 7)
        ]
        
        result = preprocessor.preprocess_batch(states)
        
        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.device.type, "cuda")
        self.assertEqual(result.shape, (2, 1, 14, 7))


class TestPreprocessingUtils(unittest.TestCase):
    """Test utility functions for preprocessing."""
    
    def setUp(self):
        """Set up test fixtures."""
        if not PREPROCESSING_AVAILABLE:
            self.skipTest("Preprocessing module not available")
    
    def test_get_state_shape(self):
        """Test get_state_shape function."""
        # Basic shape
        shape_basic = get_state_shape(include_piece_info=False)
        self.assertEqual(shape_basic, (1, 14, 7))
        
        # Enhanced shape
        shape_enhanced = get_state_shape(include_piece_info=True)
        self.assertEqual(shape_enhanced, (4, 14, 7))
        
        # Custom grid size
        shape_custom = get_state_shape(include_piece_info=True, grid_height=20, grid_width=10)
        self.assertEqual(shape_custom, (4, 20, 10))
    
    def test_validate_state_format(self):
        """Test state format validation."""
        # Valid basic state
        valid_basic = torch.rand(1, 14, 7)
        self.assertTrue(validate_state_format(valid_basic, include_piece_info=False))
        
        # Valid enhanced state
        valid_enhanced = torch.rand(4, 14, 7)
        self.assertTrue(validate_state_format(valid_enhanced, include_piece_info=True))
        
        # Invalid states
        self.assertFalse(validate_state_format(torch.rand(2, 14, 7), include_piece_info=False))  # Wrong channels
        self.assertFalse(validate_state_format(torch.rand(1, 14, 7), include_piece_info=True))   # Wrong channels
        self.assertFalse(validate_state_format(torch.rand(14, 7), include_piece_info=False))     # Wrong dimensions
        self.assertFalse(validate_state_format(np.array([1, 2, 3]), include_piece_info=False))  # Not a tensor
    
    def test_create_preprocessor_factory(self):
        """Test preprocessor factory function."""
        # Basic preprocessor
        basic_preprocessor = create_preprocessor(preprocessing_type="basic")
        self.assertTrue(callable(basic_preprocessor))
        
        # Enhanced preprocessor
        enhanced_preprocessor = create_preprocessor(preprocessing_type="enhanced")
        self.assertTrue(callable(enhanced_preprocessor))
        
        # Batch preprocessor
        batch_preprocessor = create_preprocessor(preprocessing_type="batch")
        self.assertIsInstance(batch_preprocessor, BatchPreprocessor)
        
        # Basic batch preprocessor
        batch_basic = create_preprocessor(preprocessing_type="batch_basic")
        self.assertIsInstance(batch_basic, BatchPreprocessor)
        self.assertFalse(batch_basic.include_piece_info)
        
        # Invalid type
        with self.assertRaises(ValueError):
            create_preprocessor(preprocessing_type="invalid")


class TestPreprocessingConsistency(unittest.TestCase):
    """Test consistency between different preprocessing methods."""
    
    def setUp(self):
        """Set up test fixtures."""
        if not PREPROCESSING_AVAILABLE:
            self.skipTest("Preprocessing module not available")
    
    def test_single_vs_batch_consistency(self):
        """Test that single and batch preprocessing give same results."""
        state = np.random.rand(14, 7)
        
        # Single preprocessing
        single_result = preprocess_state(state, include_piece_info=False)
        
        # Batch preprocessing with single item
        preprocessor = BatchPreprocessor(include_piece_info=False)
        batch_result = preprocessor.preprocess_batch([state])
        
        # Results should be equivalent (accounting for batch dimension)
        self.assertTrue(torch.allclose(single_result.unsqueeze(0), batch_result))
    
    def test_cpu_vs_gpu_consistency(self):
        """Test that CPU and GPU preprocessing give same results."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        
        state = np.random.rand(14, 7)
        
        # CPU preprocessing
        cpu_result = preprocess_state(state, device="cpu", include_piece_info=False)
        
        # GPU preprocessing
        gpu_result = preprocess_state(state, device="cuda", include_piece_info=False)
        
        # Results should be equal when moved to same device
        self.assertTrue(torch.allclose(cpu_result, gpu_result.cpu()))
    
    def test_different_dtypes_consistency(self):
        """Test consistency across different data types."""
        state = np.random.rand(14, 7)
        
        # Different dtypes
        result_float32 = preprocess_state(state, dtype=torch.float32, include_piece_info=False)
        result_float16 = preprocess_state(state, dtype=torch.float16, include_piece_info=False)
        
        # Should be approximately equal (float16 has less precision)
        self.assertTrue(torch.allclose(
            result_float32, 
            result_float16.float(), 
            atol=1e-3  # Allow for float16 precision loss
        ))


def run_preprocessing_tests():
    """Run all preprocessing tests and return results."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestPreprocessing))
    suite.addTests(loader.loadTestsFromTestCase(TestBatchPreprocessor))
    suite.addTests(loader.loadTestsFromTestCase(TestPreprocessingUtils))
    suite.addTests(loader.loadTestsFromTestCase(TestPreprocessingConsistency))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result


if __name__ == "__main__":
    print("Testing Preprocessing Module...")
    print("=" * 50)
    
    # Run tests
    result = run_preprocessing_tests()
    
    # Summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    
    if result.failures:
        print("\nFAILURES:")
        for test, traceback in result.failures:
            print(f"- {test}")
            print(f"  {traceback.split(chr(10))[-2] if chr(10) in traceback else traceback}")
    
    if result.errors:
        print("\nERRORS:")
        for test, traceback in result.errors:
            print(f"- {test}")
            print(f"  {traceback.split(chr(10))[-2] if chr(10) in traceback else traceback}")
    
    success = len(result.failures) == 0 and len(result.errors) == 0
    print(f"\nOverall: {'PASS' if success else 'FAIL'}")
    
    if success:
        print("✅ All preprocessing tests passed! Preprocessing is ready for use.")
    else:
        print("❌ Some preprocessing tests failed. Please fix issues before training.")
    
    exit(0 if success else 1)