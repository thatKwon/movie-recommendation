/**
 * Error handling utility functions
 */

/**
 * Custom API Error class
 */
class APIError extends Error {
  constructor(message, statusCode = 500) {
    super(message);
    this.statusCode = statusCode;
    this.name = 'APIError';
    Error.captureStackTrace(this, this.constructor);
  }
}

/**
 * Async handler wrapper to catch errors in async route handlers
 * @param {Function} fn - Async function to wrap
 * @returns {Function} Wrapped function
 */
const asyncHandler = (fn) => {
  return (req, res, next) => {
    Promise.resolve(fn(req, res, next)).catch(next);
  };
};

/**
 * Create error response object
 * @param {string} message - Error message
 * @param {number} statusCode - HTTP status code
 * @param {object} details - Additional error details
 * @returns {object} Error response object
 */

const createErrorResponse = (message, statusCode = 500, details = null) => {
  const errorResponse = {
    error: message,
    statusCode
  };

  if (details) {
    errorResponse.details = details;
  }

  if (process.env.NODE_ENV === 'development') {
    errorResponse.timestamp = new Date().toISOString();
  }

  return errorResponse;
};

/**
 * Handle Mongoose validation errors
 * @param {Error} error - Mongoose validation error
 * @returns {object} Formatted error response
 */
const handleValidationError = (error) => {
  const errors = Object.values(error.errors).map(err => err.message);
  return createErrorResponse('Validation failed', 400, errors);
};

/**
 * Handle Mongoose duplicate key errors
 * @param {Error} error - Mongoose duplicate key error
 * @returns {object} Formatted error response
 */
const handleDuplicateKeyError = (error) => {
  const field = Object.keys(error.keyPattern)[0];
  return createErrorResponse(`${field} already exists`, 400);
};

/**
 * Log error for monitoring
 * @param {Error} error - Error to log
 * @param {object} req - Express request object
 */
const logError = (error, req = null) => {
  const timestamp = new Date().toISOString();
  console.error('\n--- Error Log ---');
  console.error('Timestamp:', timestamp);
  if (req) {
    console.error('Method:', req.method);
    console.error('Path:', req.path);
    console.error('Body:', JSON.stringify(req.body, null, 2));
  }
  console.error('Error:', error.message);
  if (process.env.NODE_ENV === 'development') {
    console.error('Stack:', error.stack);
  }
  console.error('--- End Error Log ---\n');
};

module.exports = {
  APIError,
  asyncHandler,
  createErrorResponse,
  handleValidationError,
  handleDuplicateKeyError,
  logError
};
