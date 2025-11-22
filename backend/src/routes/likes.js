const express = require('express');
const router = express.Router();
const mongoose = require('mongoose');
const Like = require('../models/Like');
const Movie = require('../models/Movie');
const Actor = require('../models/Actor');
const Director = require('../models/Director');
const { authenticateToken } = require('../middleware/auth');

const getModel = (type) => {
  const models = { Movie, Actor, Director };
  return models[type];
};

// All like routes require authentication
router.use(authenticateToken);

/**
 * GET /api/likes
 * Get all items liked by the current user
 */
router.get('/', async (req, res) => {
  try {
    // UPDATED: Simplified populate.
    // This relies on "refPath: 'targetType'" being defined in the Like Schema.
    // It automatically picks the correct model (Movie, Actor, or Director).
    const userLikes = await Like.find({ userId: req.user.userId })
        .populate('targetId')
        .sort({ createdAt: -1 }); // Optional: Sort by newest likes first

    const formattedLikes = userLikes
        .filter(like => like.targetId) // Filter out likes where the target has been deleted
        .map(like => ({
          id: like._id,
          targetType: like.targetType,
          targetId: like.targetId._id,
          target: like.targetId // The populated movie/actor/director object
        }));

    res.status(200).json(formattedLikes);
  } catch (error) {
    console.error('Backend: Get all likes error:', error);
    res.status(500).json({ error: 'Server error' });
  }
});

/**
 * POST /api/likes
 * Like a movie, actor, or director
 */
router.post('/', async (req, res) => {
  console.log('Backend: POST /api/likes received');
  try {
    const { targetType, targetId } = req.body;

    if (!['Movie', 'Actor', 'Director'].includes(targetType)) {
      return res.status(400).json({ error: 'Invalid targetType' });
    }
    if (!targetId) {
      return res.status(400).json({ error: 'targetId is required' });
    }

    const Model = getModel(targetType);
    let target;

    // Handle MongoDB ID vs TMDB ID lookup
    if (mongoose.Types.ObjectId.isValid(targetId)) {
      target = await Model.findById(targetId);
    } else {
      const tmdbId = Number(targetId);
      if (!isNaN(tmdbId)) {
        target = await Model.findOne({ tmdbId });
      }
    }

    if (!target) {
      console.log('Backend: Target not found for targetId:', targetId);
      return res.status(404).json({ error: 'Target not found' });
    }

    const canonicalId = target._id;

    // Check if the like already exists
    const existingLike = await Like.findOne({
      userId: req.user.userId,
      targetType,
      targetId: canonicalId,
    });

    if (existingLike) {
      return res.status(200).json(existingLike);
    }

    // Create the new like
    const like = await Like.create({
      userId: req.user.userId,
      targetType,
      targetId: canonicalId,
    });

    console.log('Backend: Like created:', like._id);

    // Increment like count
    await Model.findByIdAndUpdate(canonicalId, { $inc: { likeCount: 1 } });

    res.status(201).json({
      userId: req.user.userId,
      targetType,
      targetId: canonicalId,
    });
  } catch (error) {
    console.error('Backend: Create like error:', error);
    res.status(500).json({ error: 'Server error' });
  }
});

/**
 * DELETE /api/likes
 * Unlike using query parameters (targetType, targetId)
 */
router.delete('/', async (req, res) => {
  console.log('Backend: DELETE /api/likes (query) received');
  try {
    const { targetType, targetId } = req.query;

    if (!['Movie', 'Actor', 'Director'].includes(targetType)) {
      return res.status(400).json({ error: 'Invalid targetType' });
    }

    if (!mongoose.Types.ObjectId.isValid(targetId)) {
      return res.status(400).json({ error: 'Invalid targetId format.' });
    }

    const deletedLike = await Like.findOneAndDelete({
      userId: req.user.userId,
      targetType,
      targetId: targetId
    });

    if (!deletedLike) {
      return res.status(404).json({ error: 'Like not found' });
    }

    console.log('Backend: Like deleted:', deletedLike._id);

    // Decrement the like count
    const Model = getModel(targetType);
    await Model.findByIdAndUpdate(targetId, { $inc: { likeCount: -1 } });

    res.status(200).json({ message: 'Like removed' });
  } catch (error) {
    console.error('Backend: Delete like by query error:', error);
    res.status(500).json({ error: 'Server error' });
  }
});

/**
 * GET /api/likes/check
 * Check if user liked multiple items
 */
router.get('/check', async (req, res) => {
  try {
    const { items } = req.query;

    if (!items) {
      return res.status(400).json({ error: 'items parameter required' });
    }

    let parsedItems;
    try {
      parsedItems = JSON.parse(items);
    } catch (error) {
      return res.status(400).json({ error: 'Invalid items format' });
    }

    const queries = parsedItems.map(item => ({
      userId: req.user.userId,
      targetType: item.type,
      targetId: item.id
    }));

    const likes = await Like.find({ $or: queries });

    // Use a Set for efficient O(1) lookups
    const likedSet = new Set(
        likes.map(like => `${like.targetType}_${like.targetId.toString()}`)
    );

    const liked = {};
    for (const item of parsedItems) {
      const key = `${item.type}_${item.id}`;
      liked[key] = likedSet.has(key);
    }

    res.status(200).json({ liked });
  } catch (error) {
    console.error('Backend: Check likes error:', error);
    res.status(500).json({ error: 'Server error' });
  }
});

module.exports = router;