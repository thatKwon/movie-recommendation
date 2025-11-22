const express = require('express');
const router = express.Router();
const User = require('../models/User');
const {
  generateAccessToken,
  generateRefreshToken,
  verifyRefreshToken,
  hashPassword,
  comparePassword
} = require('../services/authService');
const { authenticateToken } = require('../middleware/auth');

/**
 * POST /api/auth/signup
 * Register a new user
 */
router.post('/signup', async (req, res) => {
  try {
    // UPDATED: Destructure name
    const { email, password, name } = req.body;

    // Validation
    if (!email) {
      return res.status(400).json({ error: 'Email is required' });
    }

    if (!email.match(/^[^\s@]+@[^\s@]+\.[^\s@]+$/)) {
      return res.status(400).json({ error: 'Invalid email format' });
    }

    if (!password) {
      return res.status(400).json({ error: 'Password is required' });
    }

    // UPDATED: Validate name
    if (!name) {
      return res.status(400).json({ error: 'Nickname is required' });
    }

    if (password.length < 6) {
      return res.status(400).json({ error: 'Password must be at least 6 characters' });
    }

    // Check if email already exists
    const existingUser = await User.findOne({ email: email.toLowerCase() });
    if (existingUser) {
      return res.status(400).json({ error: 'Email already registered' });
    }

    // Hash password
    const hashedPassword = await hashPassword(password);

    // Create user
    const user = new User({
      email: email.toLowerCase(),
      password: hashedPassword,
      name: name, // UPDATED: Save name
      darkMode: true,
      preferredGenres: [],
      preferredActors: [],
      preferredYears: { min: 1990, max: 2024 }
    });

    await user.save();

    // Generate tokens
    const accessToken = generateAccessToken(user._id.toString(), user.email);
    const refreshToken = generateRefreshToken(user._id.toString());

    // Store refresh token in database
    user.refreshToken = refreshToken;
    await user.save();

    // Set refresh token as httpOnly cookie
    res.cookie('refreshToken', refreshToken, {
      httpOnly: true,
      secure: process.env.NODE_ENV === 'production',
      sameSite: 'strict',
      maxAge: 7 * 24 * 60 * 60 * 1000 // 7 days
    });

    // Return access token and user info
    res.status(201).json({
      accessToken,
      user: {
        id: user._id,
        email: user.email,
        name: user.name, // UPDATED: Return name
        darkMode: user.darkMode
      }
    });
  } catch (error) {
    console.error('Signup error:', error);
    res.status(500).json({ error: 'Server error' });
  }
});

/**
 * POST /api/auth/login
 * Login existing user
 */
router.post('/login', async (req, res) => {
  try {
    const { email, password } = req.body;

    // Find user (return 404 if account does not exist)
    const user = await User.findOne({ email: email.toLowerCase() });
    if (!user) {
      return res.status(404).json({ error: 'User not found' });
    }

    // Compare passwords
    const isMatch = await comparePassword(password, user.password);
    if (!isMatch) {
      return res.status(401).json({ error: 'Invalid credentials' });
    }

    // Generate new tokens
    const accessToken = generateAccessToken(user._id.toString(), user.email);
    const refreshToken = generateRefreshToken(user._id.toString());

    // Update refresh token in database
    user.refreshToken = refreshToken;
    await user.save();

    // Set refresh token cookie
    res.cookie('refreshToken', refreshToken, {
      httpOnly: true,
      secure: process.env.NODE_ENV === 'production',
      sameSite: 'strict',
      maxAge: 7 * 24 * 60 * 60 * 1000 // 7 days
    });

    // Return access token and user info
    res.status(200).json({
      accessToken,
      user: {
        id: user._id,
        email: user.email,
        name: user.name, // UPDATED: Return name
        darkMode: user.darkMode,
        preferredGenres: user.preferredGenres
      }
    });
  } catch (error) {
    console.error('Login error:', error);
    res.status(500).json({ error: 'Server error' });
  }
});

/**
 * POST /api/auth/refresh
 * Get new access token using refresh token
 */
router.post('/refresh', async (req, res) => {
  try {
    // Get refresh token from cookie
    const refreshToken = req.cookies.refreshToken;

    if (!refreshToken) {
      return res.status(401).json({ error: 'Refresh token required' });
    }

    // Verify refresh token
    let decoded;
    try {
      decoded = verifyRefreshToken(refreshToken);
    } catch (error) {
      return res.status(401).json({ error: 'Invalid refresh token' });
    }

    // Find user with matching refresh token
    const user = await User.findOne({
      _id: decoded.userId,
      refreshToken: refreshToken
    });

    if (!user) {
      return res.status(401).json({ error: 'Invalid refresh token' });
    }

    // Generate new access token
    const accessToken = generateAccessToken(user._id.toString(), user.email);

    // Return new access token
    res.status(200).json({ accessToken });
  } catch (error) {
    console.error('Refresh token error:', error);
    res.status(500).json({ error: 'Server error' });
  }
});

/**
 * POST /api/auth/logout
 * Logout user and invalidate refresh token
 * Requires authentication
 */
router.post('/logout', authenticateToken, async (req, res) => {
  try {
    // Remove refresh token from database
    await User.findByIdAndUpdate(req.user.userId, {
      refreshToken: null
    });

    // Clear refresh token cookie
    res.clearCookie('refreshToken');

    res.status(200).json({ message: 'Logged out successfully' });
  } catch (error) {
    console.error('Logout error:', error);
    res.status(500).json({ error: 'Server error' });
  }
});

module.exports = router;