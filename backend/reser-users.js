require('dotenv').config();
const mongoose = require('mongoose');
// Adjust these paths if your folder structure is different
const User = require('./src/models/User');
const Like = require('./src/models/Like');

const resetData = async () => {
    try {
        console.log('⏳ Connecting to database...');
        // Ensure we use the URI from .env or fallback to local default
        await mongoose.connect(process.env.MONGO_URI || 'mongodb://localhost:27017/movieflix');
        console.log('🔌 Connected to MongoDB.');

        // 1. Delete All Users
        const userResult = await User.deleteMany({});
        console.log(`✅ Deleted ${userResult.deletedCount} Users.`);

        // 2. Delete All Likes
        const likeResult = await Like.deleteMany({});
        console.log(`✅ Deleted ${likeResult.deletedCount} Likes.`);

        console.log('✨ Database reset complete!');
        process.exit(0);
    } catch (error) {
        console.error('❌ Error resetting database:', error);
        process.exit(1);
    }
};

resetData();
