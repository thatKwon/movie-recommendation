const { verifyAccessToken } = require('../services/authService');

/**
 * Middleware to authenticate requests using JWT access token
 * Extracts token from Authorization header, verifies it, and attaches user info to request
 */
const authenticateToken = (req, res, next) => {
  try {
    // Extract token from Authorization header
    const authHeader = req.headers['authorization'];
    const token = authHeader && authHeader.split(' ')[1]; // Format: "Bearer TOKEN"

    if (!token) {
      return res.status(401).json({ error: 'Access token required' });
    }

    // Verify token
    const decoded = verifyAccessToken(token);

    // Attach user info to request object
    req.user = {
      userId: decoded.userId,
      email: decoded.email
    };

    // Continue to next middleware/route handler
    next();
  } catch (error) {
    return res.status(401).json({ error: 'Invalid or expired token' });
  }
};

/**
 * Optional authentication middleware - doesn't fail if no token provided
 * Useful for routes that work for both authenticated and non-authenticated users
 */
const optionalAuth = (req, res, next) => {
  try {
    const authHeader = req.headers['authorization'];
    const token = authHeader && authHeader.split(' ')[1];

    if (token) {
      const decoded = verifyAccessToken(token);
      req.user = {
        userId: decoded.userId,
        email: decoded.email
      };
    }
  } catch (error) {
    // Token is invalid, but we don't fail the request
    // Just continue without setting req.user
  }

  next();
};

module.exports = {
  authenticateToken,
  optionalAuth
};
