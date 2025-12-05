// Zubr Website JavaScript
// Mobile navigation toggle and smooth scrolling

document.addEventListener('DOMContentLoaded', function() {
    // Mobile Navigation Toggle
    const hamburger = document.querySelector('.hamburger');
    const nav = document.querySelector('.nav');
    
    if (hamburger && nav) {
        hamburger.addEventListener('click', function() {
            hamburger.classList.toggle('active');
            nav.classList.toggle('active');
            
            // Update ARIA attributes
            const isExpanded = hamburger.classList.contains('active');
            hamburger.setAttribute('aria-expanded', isExpanded);
            nav.setAttribute('aria-hidden', !isExpanded);
        });
        
        // Close menu when clicking a link
        const navLinks = nav.querySelectorAll('.nav-link');
        navLinks.forEach(function(link) {
            link.addEventListener('click', function() {
                hamburger.classList.remove('active');
                nav.classList.remove('active');
                hamburger.setAttribute('aria-expanded', 'false');
                nav.setAttribute('aria-hidden', 'true');
            });
        });
        
        // Close menu when clicking outside
        document.addEventListener('click', function(event) {
            if (!nav.contains(event.target) && !hamburger.contains(event.target)) {
                hamburger.classList.remove('active');
                nav.classList.remove('active');
                hamburger.setAttribute('aria-expanded', 'false');
                nav.setAttribute('aria-hidden', 'true');
            }
        });
    }
    
    // FAQ Accordion
    const faqItems = document.querySelectorAll('.faq-item');
    
    faqItems.forEach(function(item) {
        const question = item.querySelector('.faq-question');
        
        if (question) {
            question.addEventListener('click', function() {
                // Close other items
                faqItems.forEach(function(otherItem) {
                    if (otherItem !== item) {
                        otherItem.classList.remove('active');
                    }
                });
                
                // Toggle current item
                item.classList.toggle('active');
            });
        }
    });
    
    // Smooth scrolling for anchor links
    const anchorLinks = document.querySelectorAll('a[href^="#"]');
    
    anchorLinks.forEach(function(link) {
        link.addEventListener('click', function(e) {
            const targetId = this.getAttribute('href');
            
            if (targetId !== '#') {
                const targetElement = document.querySelector(targetId);
                
                if (targetElement) {
                    e.preventDefault();
                    targetElement.scrollIntoView({
                        behavior: 'smooth',
                        block: 'start'
                    });
                }
            }
        });
    });
    
    // Set current year in footer
    const yearElement = document.querySelector('.current-year');
    if (yearElement) {
        yearElement.textContent = new Date().getFullYear();
    }
    
    // Mark active navigation link
    const currentPage = window.location.pathname.split('/').pop() || 'index.html';
    const navLinks2 = document.querySelectorAll('.nav-link');
    
    navLinks2.forEach(function(link) {
        const href = link.getAttribute('href');
        if (href === currentPage || (currentPage === '' && href === 'index.html')) {
            link.classList.add('active');
        }
    });
});
