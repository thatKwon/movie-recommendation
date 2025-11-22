'use client';

import React, { createContext, useContext, useState, useEffect, ReactNode } from 'react';
import { useAuth } from './AuthContext';

interface ThemeContextType {
  darkMode: boolean;
  toggleDarkMode: () => void;
}

const ThemeContext = createContext<ThemeContextType | undefined>(undefined);

export const ThemeProvider = ({ children }: { children: ReactNode }) => {
  const { user, accessToken } = useAuth();
  const [darkMode, setDarkMode] = useState(true); // Default to dark mode

  // Sync with user preferences
  useEffect(() => {
    if (user && user.darkMode !== undefined) {
      setDarkMode(user.darkMode);
    }
  }, [user]);

  // Apply theme classes to html element
  useEffect(() => {
    const el = document.documentElement;
    if (darkMode) {
      el.classList.add('dark');
      el.classList.remove('light');
    } else {
      el.classList.remove('dark');
      el.classList.add('light');
    }
  }, [darkMode]);

  const toggleDarkMode = async () => {
    const newDarkMode = !darkMode;
    setDarkMode(newDarkMode);

    // Update on server if authenticated
    if (accessToken) {
      try {
        await fetch(`${process.env.NEXT_PUBLIC_API_URL}/api/user/profile`, {
          method: 'PATCH',
          headers: {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${accessToken}`
          },
          credentials: 'include',
          body: JSON.stringify({ darkMode: newDarkMode })
        });
      } catch (error) {
        console.error('Failed to update dark mode preference:', error);
      }
    }
  };

  return (
    <ThemeContext.Provider value={{ darkMode, toggleDarkMode }}>
      {children}
    </ThemeContext.Provider>
  );
};

export const useTheme = () => {
  const context = useContext(ThemeContext);
  if (context === undefined) {
    throw new Error('useTheme must be used within a ThemeProvider');
  }
  return context;
};
