"""
Unit tests for AWS-related functions in ask_bedrock.main.
"""

import unittest
from unittest.mock import patch, MagicMock

from ask_bedrock.main import get_bedrock_runtime, get_bedrock


class TestAWSFunctions(unittest.TestCase):
    """Test the AWS integration functions."""

    @patch("ask_bedrock.main.boto3.Session")
    def test_get_bedrock_runtime(self, mock_session):
        """Test creating a bedrock-runtime client with the correct configuration."""
        # Setup
        mock_client = MagicMock()
        mock_session_instance = MagicMock()
        mock_session_instance.client.return_value = mock_client
        mock_session.return_value = mock_session_instance
        
        config = {
            "aws_profile": "test-profile",
            "region": "us-west-2"
        }
        
        # Execute
        result = get_bedrock_runtime(config)
        
        # Assert
        mock_session.assert_called_once_with(profile_name="test-profile")
        mock_session_instance.client.assert_called_once_with("bedrock-runtime", "us-west-2")
        self.assertEqual(result, mock_client)
    
    @patch("ask_bedrock.main.boto3.Session")
    def test_get_bedrock(self, mock_session):
        """Test creating a bedrock client with the correct configuration."""
        # Setup
        mock_client = MagicMock()
        mock_session_instance = MagicMock()
        mock_session_instance.client.return_value = mock_client
        mock_session.return_value = mock_session_instance
        
        config = {
            "aws_profile": "test-profile",
            "region": "us-west-2"
        }
        
        # Execute
        result = get_bedrock(config)
        
        # Assert
        mock_session.assert_called_once_with(profile_name="test-profile")
        mock_session_instance.client.assert_called_once_with("bedrock", "us-west-2")
        self.assertEqual(result, mock_client)


if __name__ == "__main__":
    unittest.main()