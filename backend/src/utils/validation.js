/**
 * Validation utility functions
 */

/**
 * Validate email format
 * @param {string} email - Email address to validate
 * @returns {boolean} True if valid
 */
const isValidEmail = (email) => {
  const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
  return emailRegex.test(email);
};

/**
 * Validate password strength
 * @param {string} password - Password to validate
 * @returns {object} { valid: boolean, message: string }
 */
const validatePassword = (password) => {
  if (!password) {
    return { valid: false, message: 'Password is required' };
  }

  if (password.length < 6) {
    return { valid: false, message: 'Password must be at least 6 characters' };
  }

  return { valid: true, message: 'Password is valid' };
};

/**
 * Validate MongoDB ObjectId format
 * @param {string} id - ID to validate
 * @returns {boolean} True if valid ObjectId format
 */
const isValidObjectId = (id) => {
  const objectIdRegex = /^[a-f\d]{24}$/i;
  return objectIdRegex.test(id);
};

/**
 * Validate genre selection
 * @param {Array} genres - Array of genre strings
 * @returns {object} { valid: boolean, message: string }
 */
const validateGenres = (genres) => {
  if (!Array.isArray(genres)) {
    return { valid: false, message: 'Genres must be an array' };
  }

  if (genres.length === 0) {
    return { valid: false, message: 'At least one genre is required' };
  }

  return { valid: true, message: 'Genres are valid' };
};

/**
 * Validate year range
 * @param {object} yearRange - Object with min and max properties
 * @returns {object} { valid: boolean, message: string }
 */
const validateYearRange = (yearRange) => {
  if (!yearRange || typeof yearRange !== 'object') {
    return { valid: false, message: 'Year range must be an object' };
  }

  const { min, max } = yearRange;

  if (typeof min !== 'number' || typeof max !== 'number') {
    return { valid: false, message: 'Year range min and max must be numbers' };
  }

  if (min < 1900 || max > new Date().getFullYear() + 1) {
    return { valid: false, message: 'Year range is out of valid bounds' };
  }

  if (min > max) {
    return { valid: false, message: 'Minimum year cannot be greater than maximum year' };
  }

  return { valid: true, message: 'Year range is valid' };
};

/**
 * Sanitize string input
 * @param {string} input - String to sanitize
 * @returns {string} Sanitized string
 */
const sanitizeString = (input) => {
  if (typeof input !== 'string') return '';
  return input.trim();
};

module.exports = {
  isValidEmail,
  validatePassword,
  isValidObjectId,
  validateGenres,
  validateYearRange,
  sanitizeString
};
