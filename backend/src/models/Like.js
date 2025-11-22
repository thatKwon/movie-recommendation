const mongoose = require('mongoose');

const likeSchema = new mongoose.Schema({
  userId: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'User',
    required: true
  },
  targetType: {
    type: String,
    required: true,
    enum: ['Movie', 'Actor', 'Director'] // Capitalized to match model names
  },
  targetId: {
    type: mongoose.Schema.Types.ObjectId,
    required: true,
    refPath: 'targetType' // Dynamic reference
  }
}, {
  timestamps: true
});

// Create compound unique index to prevent duplicate likes
likeSchema.index({ userId: 1, targetType: 1, targetId: 1 }, { unique: true });
likeSchema.index({ userId: 1 });
likeSchema.index({ targetId: 1, targetType: 1 });

module.exports = mongoose.model('Like', likeSchema);
