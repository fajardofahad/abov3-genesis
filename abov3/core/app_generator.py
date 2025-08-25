"""
ABOV3 Genesis - Full Application Generator
Creates complete, production-ready applications from high-level descriptions
"""

import asyncio
import os
import json
import uuid
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from .code_generator import CodeGenerator

class FullApplicationGenerator:
    """
    Full Application Generator for ABOV3 Genesis
    Creates complete applications with all necessary components
    """
    
    def __init__(self, project_path: Path):
        self.project_path = Path(project_path)
        self.code_generator = CodeGenerator(project_path)
        
        # Application templates and configurations
        self.app_templates = self._initialize_templates()
        
    def _initialize_templates(self) -> Dict[str, Any]:
        """Initialize application templates and configurations"""
        return {
            'website_templates': {
                'business': {
                    'pages': ['home', 'about', 'services', 'contact'],
                    'features': ['responsive_design', 'contact_form', 'gallery', 'testimonials'],
                    'tech_stack': 'html_css_js'
                },
                'ecommerce': {
                    'pages': ['home', 'products', 'product_detail', 'cart', 'checkout', 'account'],
                    'features': ['shopping_cart', 'payment_integration', 'user_auth', 'admin_panel'],
                    'tech_stack': 'react_node_mongodb'
                },
                'restaurant': {
                    'pages': ['home', 'menu', 'about', 'contact', 'reservations'],
                    'features': ['menu_display', 'online_ordering', 'reservation_system'],
                    'tech_stack': 'html_css_js_backend'
                },
                'portfolio': {
                    'pages': ['home', 'projects', 'about', 'contact'],
                    'features': ['project_gallery', 'contact_form', 'resume_download'],
                    'tech_stack': 'html_css_js'
                }
            },
            'mobile_templates': {
                'flutter': {
                    'structure': ['lib/main.dart', 'lib/screens/', 'lib/widgets/', 'lib/models/'],
                    'features': ['navigation', 'state_management', 'api_integration']
                },
                'react_native': {
                    'structure': ['App.js', 'src/screens/', 'src/components/', 'src/services/'],
                    'features': ['navigation', 'redux', 'api_calls']
                }
            }
        }
    
    async def generate_full_application(self, description: str, preferences: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Generate a complete application from description
        
        Args:
            description: High-level description of the application
            preferences: User preferences for tech stack, features, etc.
        
        Returns:
            Dict with generation results
        """
        print(f"[DEBUG] Generating full application from: {description[:100]}...")
        
        try:
            # Step 1: Analyze the requirement
            analysis = await self._analyze_application_requirement(description, preferences)
            
            # Step 2: Plan the application architecture
            architecture = await self._plan_application_architecture(analysis)
            
            # Step 3: Generate all components
            generation_result = await self._generate_application_components(architecture)
            
            # Step 4: Install dependencies and setup environment
            setup_result = await self._setup_development_environment(architecture)
            
            # Step 5: Create deployment configurations (only if requested)
            deployment_config = {}
            if self._should_create_deployment_config(description, preferences):
                deployment_config = await self._create_deployment_config(architecture)
            
            # Step 6: Generate documentation
            documentation = await self._generate_documentation(architecture, generation_result)
            
            return {
                'success': True,
                'analysis': analysis,
                'architecture': architecture,
                'generated_files': generation_result.get('files', []),
                'setup_result': setup_result,
                'deployment': deployment_config,
                'documentation': documentation,
                'next_steps': self._get_next_steps(architecture)
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'generated_files': []
            }
    
    async def _analyze_application_requirement(self, description: str, preferences: Dict[str, Any] = None) -> Dict[str, Any]:
        """Analyze the application requirement and determine type, features, etc."""
        analysis = {
            'app_type': 'unknown',
            'category': 'unknown',
            'features': [],
            'complexity': 'medium',
            'tech_stack': 'recommended',
            'target_platform': 'web',
            'estimated_files': 10,
            'estimated_components': []
        }
        
        description_lower = description.lower()
        
        # Determine application type
        if any(word in description_lower for word in ['website', 'web app', 'web application', 'site']):
            analysis['target_platform'] = 'web'
            
            # Determine website category (prioritize restaurant/cafe over ecommerce)
            if any(word in description_lower for word in ['restaurant', 'cafe', 'coffee', 'food', 'menu', 'dining']):
                analysis['app_type'] = 'restaurant'
                analysis['category'] = 'business'
                analysis['complexity'] = 'medium'
                analysis['estimated_files'] = 15
            elif any(word in description_lower for word in ['shop', 'store', 'ecommerce', 'e-commerce', 'buy', 'sell']) and not any(word in description_lower for word in ['coffee shop', 'restaurant', 'cafe', 'food']):
                analysis['app_type'] = 'ecommerce'
                analysis['category'] = 'business'
                analysis['complexity'] = 'high'
                analysis['estimated_files'] = 25
            elif any(word in description_lower for word in ['portfolio', 'personal', 'resume', 'cv']):
                analysis['app_type'] = 'portfolio'
                analysis['category'] = 'personal'
                analysis['complexity'] = 'low'
                analysis['estimated_files'] = 8
            elif any(word in description_lower for word in ['blog', 'news', 'article', 'content']):
                analysis['app_type'] = 'blog'
                analysis['category'] = 'content'
                analysis['complexity'] = 'medium'
                analysis['estimated_files'] = 12
            else:
                analysis['app_type'] = 'business'
                analysis['category'] = 'business'
                analysis['complexity'] = 'medium'
                analysis['estimated_files'] = 10
                
        elif any(word in description_lower for word in ['mobile app', 'mobile application', 'android', 'ios', 'phone app']):
            analysis['target_platform'] = 'mobile'
            analysis['app_type'] = 'mobile_app'
            analysis['complexity'] = 'high'
            analysis['estimated_files'] = 20
            
        elif any(word in description_lower for word in ['api', 'backend', 'service', 'microservice']):
            analysis['target_platform'] = 'backend'
            analysis['app_type'] = 'api'
            analysis['complexity'] = 'medium'
            analysis['estimated_files'] = 12
            
        # Extract features from description
        feature_keywords = {
            'user_auth': ['login', 'register', 'authentication', 'user account', 'sign up', 'sign in'],
            'shopping_cart': ['cart', 'shopping', 'add to cart', 'checkout', 'purchase'],
            'payment': ['payment', 'pay', 'billing', 'credit card', 'stripe', 'paypal'],
            'admin_panel': ['admin', 'dashboard', 'manage', 'administration'],
            'search': ['search', 'find', 'filter', 'query'],
            'comments': ['comment', 'review', 'feedback', 'rating'],
            'notifications': ['notification', 'alert', 'email', 'sms'],
            'responsive_design': ['mobile', 'responsive', 'device', 'phone', 'tablet'],
            'database': ['data', 'store', 'save', 'database', 'records'],
            'api_integration': ['api', 'integration', 'third party', 'external']
        }
        
        for feature, keywords in feature_keywords.items():
            if any(keyword in description_lower for keyword in keywords):
                analysis['features'].append(feature)
        
        # Determine recommended tech stack
        if analysis['target_platform'] == 'web':
            # For restaurants and portfolios, prefer simple static websites unless complex features are needed
            if analysis['app_type'] in ['restaurant', 'portfolio'] and not any(feature in analysis['features'] for feature in ['shopping_cart', 'user_auth', 'admin_panel']):
                analysis['tech_stack'] = 'html_css_js'
            elif analysis['complexity'] == 'high' or 'shopping_cart' in analysis['features']:
                analysis['tech_stack'] = 'react_node_mongodb'
            elif analysis['complexity'] == 'medium':
                analysis['tech_stack'] = 'html_css_js_backend'
            else:
                analysis['tech_stack'] = 'html_css_js'
        elif analysis['target_platform'] == 'mobile':
            analysis['tech_stack'] = 'flutter'  # Default to Flutter for cross-platform
        elif analysis['target_platform'] == 'backend':
            analysis['tech_stack'] = 'fastapi_python'
        
        # Apply user preferences
        if preferences:
            if preferences.get('tech_stack'):
                analysis['tech_stack'] = preferences['tech_stack']
            if preferences.get('features'):
                analysis['features'].extend(preferences['features'])
        
        return analysis
    
    async def _plan_application_architecture(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Plan the application architecture based on analysis"""
        architecture = {
            'app_type': analysis['app_type'],
            'tech_stack': analysis['tech_stack'],
            'target_platform': analysis['target_platform'],
            'components': [],
            'file_structure': {},
            'dependencies': {},
            'database_schema': {},
            'api_endpoints': [],
            'pages': [],
            'features': analysis['features']
        }
        
        # Plan based on application type
        if analysis['app_type'] == 'ecommerce':
            architecture.update(await self._plan_ecommerce_architecture(analysis))
        elif analysis['app_type'] == 'restaurant':
            architecture.update(await self._plan_restaurant_architecture(analysis))
        elif analysis['app_type'] == 'portfolio':
            architecture.update(await self._plan_portfolio_architecture(analysis))
        elif analysis['app_type'] == 'mobile_app':
            architecture.update(await self._plan_mobile_architecture(analysis))
        elif analysis['app_type'] == 'api':
            architecture.update(await self._plan_api_architecture(analysis))
        else:
            architecture.update(await self._plan_business_website_architecture(analysis))
        
        return architecture
    
    async def _plan_ecommerce_architecture(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Plan ecommerce application architecture"""
        return {
            'pages': ['home', 'products', 'product-detail', 'cart', 'checkout', 'account', 'admin'],
            'components': [
                'ProductCard', 'CartItem', 'Navigation', 'Footer', 'SearchBar',
                'PaymentForm', 'UserProfile', 'AdminDashboard', 'ProductForm'
            ],
            'file_structure': {
                'frontend': {
                    'src/pages/': ['HomePage.js', 'ProductsPage.js', 'ProductDetailPage.js', 'CartPage.js', 'CheckoutPage.js'],
                    'src/components/': ['ProductCard.js', 'Navigation.js', 'Footer.js', 'CartItem.js'],
                    'src/contexts/': ['CartContext.js', 'AuthContext.js'],
                    'src/services/': ['api.js', 'auth.js'],
                    'src/styles/': ['global.css', 'components.css']
                },
                'backend': {
                    'routes/': ['products.js', 'cart.js', 'auth.js', 'orders.js'],
                    'models/': ['Product.js', 'User.js', 'Order.js'],
                    'middleware/': ['auth.js', 'validation.js'],
                    'controllers/': ['productController.js', 'userController.js']
                }
            },
            'dependencies': {
                'frontend': ['react', 'react-router-dom', 'axios', 'react-context-api', '@stripe/stripe-js'],
                'backend': ['express', 'mongoose', 'bcryptjs', 'jsonwebtoken', 'stripe', 'cors']
            },
            'database_schema': {
                'products': ['id', 'name', 'description', 'price', 'image', 'category', 'stock'],
                'users': ['id', 'email', 'password', 'name', 'address', 'role'],
                'orders': ['id', 'user_id', 'products', 'total', 'status', 'created_at']
            },
            'api_endpoints': [
                'GET /api/products', 'GET /api/products/:id', 'POST /api/cart/add',
                'POST /api/auth/login', 'POST /api/auth/register', 'POST /api/orders'
            ]
        }
    
    async def _plan_restaurant_architecture(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Plan restaurant website architecture"""
        return {
            'pages': ['home', 'menu', 'about', 'contact', 'reservations', 'order'],
            'components': [
                'MenuItem', 'Navigation', 'Footer', 'ReservationForm', 
                'OrderCart', 'MenuCategory', 'ContactForm'
            ],
            'file_structure': {
                'src/pages/': ['HomePage.js', 'MenuPage.js', 'AboutPage.js', 'ContactPage.js', 'ReservationsPage.js'],
                'src/components/': ['MenuItem.js', 'Navigation.js', 'Footer.js', 'ReservationForm.js'],
                'src/data/': ['menu.js', 'restaurant-info.js'],
                'src/styles/': ['global.css', 'components.css', 'pages.css'],
                'assets/images/': ['logo.png', 'hero-bg.jpg', 'menu-items/']
            },
            'dependencies': {
                'frontend': ['react', 'react-router-dom', 'react-datepicker', 'emailjs-com']
            },
            'database_schema': {
                'menu_items': ['id', 'name', 'description', 'price', 'category', 'image', 'available'],
                'reservations': ['id', 'name', 'email', 'phone', 'date', 'time', 'party_size', 'special_requests'],
                'orders': ['id', 'customer_info', 'items', 'total', 'status', 'order_type']
            }
        }
    
    async def _plan_portfolio_architecture(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Plan portfolio website architecture"""
        return {
            'pages': ['home', 'projects', 'about', 'contact'],
            'components': ['ProjectCard', 'Navigation', 'Footer', 'ContactForm', 'SkillsList'],
            'file_structure': {
                'src/pages/': ['HomePage.js', 'ProjectsPage.js', 'AboutPage.js', 'ContactPage.js'],
                'src/components/': ['ProjectCard.js', 'Navigation.js', 'Footer.js', 'ContactForm.js'],
                'src/data/': ['projects.js', 'skills.js', 'personal-info.js'],
                'src/styles/': ['global.css', 'components.css', 'animations.css'],
                'assets/': ['images/', 'resume.pdf', 'projects/']
            },
            'dependencies': {
                'frontend': ['react', 'react-router-dom', 'framer-motion', 'emailjs-com']
            }
        }
    
    async def _plan_mobile_architecture(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Plan mobile application architecture"""
        return {
            'pages': ['splash', 'home', 'profile', 'settings'],
            'components': ['CustomButton', 'Header', 'Card', 'LoadingSpinner'],
            'file_structure': {
                'lib/': ['main.dart'],
                'lib/screens/': ['splash_screen.dart', 'home_screen.dart', 'profile_screen.dart'],
                'lib/widgets/': ['custom_button.dart', 'card_widget.dart', 'header_widget.dart'],
                'lib/models/': ['user_model.dart', 'app_data.dart'],
                'lib/services/': ['api_service.dart', 'storage_service.dart'],
                'lib/utils/': ['constants.dart', 'helpers.dart'],
                'assets/': ['images/', 'fonts/']
            },
            'dependencies': {
                'flutter': ['http', 'shared_preferences', 'provider', 'flutter_svg', 'cached_network_image']
            }
        }
    
    async def _plan_api_architecture(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Plan API/backend architecture"""
        return {
            'api_endpoints': ['GET /api/health', 'POST /api/auth/login', 'GET /api/data'],
            'components': ['AuthController', 'DataController', 'ValidationMiddleware'],
            'file_structure': {
                'app/': ['main.py'],
                'app/routes/': ['auth.py', 'data.py'],
                'app/models/': ['user.py', 'data_model.py'],
                'app/services/': ['auth_service.py', 'data_service.py'],
                'app/utils/': ['database.py', 'security.py'],
                'tests/': ['test_auth.py', 'test_api.py']
            },
            'dependencies': {
                'python': ['fastapi', 'uvicorn', 'sqlalchemy', 'pydantic', 'python-jose', 'passlib']
            }
        }
    
    async def _plan_business_website_architecture(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Plan general business website architecture"""
        return {
            'pages': ['home', 'about', 'services', 'contact'],
            'components': ['Navigation', 'Footer', 'ServiceCard', 'ContactForm', 'Hero'],
            'file_structure': {
                'src/pages/': ['HomePage.js', 'AboutPage.js', 'ServicesPage.js', 'ContactPage.js'],
                'src/components/': ['Navigation.js', 'Footer.js', 'ServiceCard.js', 'ContactForm.js'],
                'src/styles/': ['global.css', 'components.css'],
                'assets/': ['images/', 'icons/']
            },
            'dependencies': {
                'frontend': ['react', 'react-router-dom', 'emailjs-com']
            }
        }
    
    async def _generate_application_components(self, architecture: Dict[str, Any]) -> Dict[str, Any]:
        """Generate all application components based on architecture"""
        generation_result = {
            'files': [],
            'components_created': 0,
            'pages_created': 0,
            'total_lines': 0
        }
        
        try:
            # Generate based on tech stack
            tech_stack = architecture.get('tech_stack', 'html_css_js')
            
            if tech_stack == 'react_node_mongodb':
                await self._generate_react_app(architecture, generation_result)
            elif tech_stack == 'html_css_js_backend':
                await self._generate_html_js_app_with_backend(architecture, generation_result)
            elif tech_stack == 'html_css_js':
                await self._generate_static_website(architecture, generation_result)
            elif tech_stack == 'flutter':
                await self._generate_flutter_app(architecture, generation_result)
            elif tech_stack == 'fastapi_python':
                await self._generate_fastapi_backend(architecture, generation_result)
            else:
                # Default to static website
                await self._generate_static_website(architecture, generation_result)
            
            return generation_result
            
        except Exception as e:
            print(f"[DEBUG] Error generating application components: {e}")
            return generation_result
    
    async def _generate_react_app(self, architecture: Dict[str, Any], result: Dict[str, Any]):
        """Generate complete React + Node.js + MongoDB application"""
        app_type = architecture.get('app_type', 'business')
        
        # Generate package.json for frontend
        frontend_package = {
            "name": f"{app_type}-frontend",
            "version": "1.0.0",
            "private": True,
            "dependencies": {
                "react": "^18.2.0",
                "react-dom": "^18.2.0",
                "react-router-dom": "^6.8.0",
                "axios": "^1.3.0",
                "@testing-library/jest-dom": "^5.16.4",
                "@testing-library/react": "^13.4.0",
                "@testing-library/user-event": "^13.5.0",
                "web-vitals": "^2.1.4"
            },
            "scripts": {
                "start": "react-scripts start",
                "build": "react-scripts build",
                "test": "react-scripts test",
                "eject": "react-scripts eject"
            },
            "eslintConfig": {
                "extends": ["react-app", "react-app/jest"]
            },
            "browserslist": {
                "production": [">0.2%", "not dead", "not op_mini all"],
                "development": ["last 1 chrome version", "last 1 firefox version", "last 1 safari version"]
            }
        }
        
        # Add specific dependencies based on features
        if 'shopping_cart' in architecture.get('features', []):
            frontend_package['dependencies']['@stripe/stripe-js'] = '^1.46.0'
        
        file_result = await self.code_generator.create_file(
            'frontend/package.json',
            json.dumps(frontend_package, indent=2),
            'Frontend package configuration'
        )
        if file_result.get('success'):
            result['files'].append(file_result)
        
        # Generate main App.js
        app_js_content = self._generate_react_app_component(architecture)
        file_result = await self.code_generator.create_file('frontend/src/App.js', app_js_content, 'Main React App component')
        if file_result.get('success'):
            result['files'].append(file_result)
        
        # Generate pages
        for page in architecture.get('pages', []):
            page_content = self._generate_react_page(page, architecture)
            page_name = f"{page.replace('-', '_').title().replace('_', '')}Page"
            file_result = await self.code_generator.create_file(
                f'frontend/src/pages/{page_name}.js',
                page_content,
                f'{page_name} component'
            )
            if file_result.get('success'):
                result['files'].append(file_result)
                result['pages_created'] += 1
        
        # Generate components
        for component in architecture.get('components', []):
            component_content = self._generate_react_component(component, architecture)
            file_result = await self.code_generator.create_file(
                f'frontend/src/components/{component}.js',
                component_content,
                f'{component} React component'
            )
            if file_result.get('success'):
                result['files'].append(file_result)
                result['components_created'] += 1
        
        # Generate backend package.json and server
        await self._generate_node_backend(architecture, result)
        
        # Generate styling
        await self._generate_react_styles(architecture)
        
        # Update results
        result['files'].extend([
            'frontend/package.json', 'frontend/src/App.js', 'backend/package.json', 'backend/server.js'
        ])
    
    def _generate_react_app_component(self, architecture: Dict[str, Any]) -> str:
        """Generate main React App component"""
        pages = architecture.get('pages', [])
        imports = []
        routes = []
        
        for page in pages:
            page_name = f"{page.replace('-', '_').title().replace('_', '')}Page"
            imports.append(f"import {page_name} from './pages/{page_name}';")
            
            if page == 'home':
                routes.append(f'        <Route path="/" element={{{page_name}()}} />')
            else:
                routes.append(f'        <Route path="/{page}" element={{{page_name}()}} />')
        
        return f"""import React from 'react';
import {{ BrowserRouter as Router, Routes, Route }} from 'react-router-dom';
{chr(10).join(imports)}
import Navigation from './components/Navigation';
import Footer from './components/Footer';
import './styles/global.css';

function App() {{
  return (
    <Router>
      <div className="App">
        <Navigation />
        <main className="main-content">
          <Routes>
{chr(10).join(routes)}
          </Routes>
        </main>
        <Footer />
      </div>
    </Router>
  );
}}

export default App;"""
    
    def _generate_react_page(self, page: str, architecture: Dict[str, Any]) -> str:
        """Generate React page components"""
        app_type = architecture.get('app_type', 'business')
        page_name = f"{page.replace('-', '_').title().replace('_', '')}Page"
        
        if page == 'home':
            return self._generate_home_page_react(app_type)
        elif page == 'products':
            return self._generate_products_page_react()
        elif page == 'cart':
            return self._generate_cart_page_react()
        elif page == 'menu':
            return self._generate_menu_page_react()
        elif page == 'about':
            return self._generate_about_page_react(app_type)
        elif page == 'contact':
            return self._generate_contact_page_react()
        else:
            return f"""import React from 'react';

const {page_name} = () => {{
  return (
    <div className="{page}-page">
      <div className="container">
        <h1>{page.replace('-', ' ').title()}</h1>
        <p>Welcome to the {page.replace('-', ' ')} page.</p>
      </div>
    </div>
  );
}};

export default {page_name};"""
    
    def _generate_home_page_react(self, app_type: str) -> str:
        """Generate home page for React app"""
        if app_type == 'ecommerce':
            return """import React from 'react';
import { Link } from 'react-router-dom';

const HomePage = () => {
  return (
    <div className="home-page">
      {/* Hero Section */}
      <section className="hero">
        <div className="container">
          <div className="hero-content">
            <h1>Welcome to Our Store</h1>
            <p>Discover amazing products at unbeatable prices</p>
            <Link to="/products" className="cta-button">
              Shop Now
            </Link>
          </div>
        </div>
      </section>

      {/* Featured Products */}
      <section className="featured-products">
        <div className="container">
          <h2>Featured Products</h2>
          <div className="products-grid">
            {/* Product cards will be populated from API */}
          </div>
        </div>
      </section>

      {/* About Section */}
      <section className="about-preview">
        <div className="container">
          <h2>Why Choose Us?</h2>
          <div className="features">
            <div className="feature">
              <h3>Quality Products</h3>
              <p>We offer only the highest quality products</p>
            </div>
            <div className="feature">
              <h3>Fast Shipping</h3>
              <p>Quick and reliable delivery to your door</p>
            </div>
            <div className="feature">
              <h3>Great Support</h3>
              <p>24/7 customer support for all your needs</p>
            </div>
          </div>
        </div>
      </section>
    </div>
  );
};

export default HomePage;"""
        elif app_type == 'restaurant':
            return """import React from 'react';
import { Link } from 'react-router-dom';

const HomePage = () => {
  return (
    <div className="home-page">
      {/* Hero Section */}
      <section className="hero">
        <div className="hero-content">
          <h1>Welcome to Our Restaurant</h1>
          <p>Experience exceptional dining with fresh, locally sourced ingredients</p>
          <div className="hero-buttons">
            <Link to="/menu" className="cta-button primary">
              View Menu
            </Link>
            <Link to="/reservations" className="cta-button secondary">
              Make Reservation
            </Link>
          </div>
        </div>
      </section>

      {/* Features */}
      <section className="features">
        <div className="container">
          <h2>Why Dine With Us</h2>
          <div className="features-grid">
            <div className="feature">
              <h3>Fresh Ingredients</h3>
              <p>We source the freshest ingredients daily</p>
            </div>
            <div className="feature">
              <h3>Expert Chefs</h3>
              <p>Our experienced chefs create culinary masterpieces</p>
            </div>
            <div className="feature">
              <h3>Cozy Atmosphere</h3>
              <p>Perfect ambiance for any occasion</p>
            </div>
          </div>
        </div>
      </section>

      {/* Menu Preview */}
      <section className="menu-preview">
        <div className="container">
          <h2>Popular Dishes</h2>
          <div className="menu-items">
            {/* Menu items will be populated */}
          </div>
          <Link to="/menu" className="view-full-menu">View Full Menu</Link>
        </div>
      </section>
    </div>
  );
};

export default HomePage;"""
        else:
            return """import React from 'react';
import { Link } from 'react-router-dom';

const HomePage = () => {
  return (
    <div className="home-page">
      {/* Hero Section */}
      <section className="hero">
        <div className="container">
          <div className="hero-content">
            <h1>Welcome to Our Business</h1>
            <p>Your trusted partner for exceptional services and solutions</p>
            <Link to="/services" className="cta-button">
              Learn More
            </Link>
          </div>
        </div>
      </section>

      {/* Services Preview */}
      <section className="services-preview">
        <div className="container">
          <h2>Our Services</h2>
          <div className="services-grid">
            <div className="service">
              <h3>Service One</h3>
              <p>Description of your first service</p>
            </div>
            <div className="service">
              <h3>Service Two</h3>
              <p>Description of your second service</p>
            </div>
            <div className="service">
              <h3>Service Three</h3>
              <p>Description of your third service</p>
            </div>
          </div>
        </div>
      </section>
    </div>
  );
};

export default HomePage;"""
    
    def _generate_react_component(self, component: str, architecture: Dict[str, Any]) -> str:
        """Generate React components"""
        if component == 'Navigation':
            pages = architecture.get('pages', [])
            nav_links = []
            
            for page in pages:
                if page == 'home':
                    nav_links.append('        <Link to="/" className="nav-link">Home</Link>')
                else:
                    nav_links.append(f'        <Link to="/{page}" className="nav-link">{page.replace("-", " ").title()}</Link>')
            
            return f"""import React from 'react';
import {{ Link }} from 'react-router-dom';

const Navigation = () => {{
  return (
    <nav className="navigation">
      <div className="container">
        <Link to="/" className="logo">
          Your Brand
        </Link>
        <div className="nav-links">
{chr(10).join(nav_links)}
        </div>
      </div>
    </nav>
  );
}};

export default Navigation;"""
            
        elif component == 'Footer':
            return """import React from 'react';

const Footer = () => {
  return (
    <footer className="footer">
      <div className="container">
        <div className="footer-content">
          <div className="footer-section">
            <h3>Your Brand</h3>
            <p>Your business description goes here.</p>
          </div>
          <div className="footer-section">
            <h4>Quick Links</h4>
            <ul>
              <li><a href="/">Home</a></li>
              <li><a href="/about">About</a></li>
              <li><a href="/contact">Contact</a></li>
            </ul>
          </div>
          <div className="footer-section">
            <h4>Contact Info</h4>
            <p>Email: info@yourbusiness.com</p>
            <p>Phone: (555) 123-4567</p>
          </div>
        </div>
        <div className="footer-bottom">
          <p>&copy; 2024 Your Brand. All rights reserved.</p>
        </div>
      </div>
    </footer>
  );
};

export default Footer;"""
        
        else:
            return f"""import React from 'react';

const {component} = () => {{
  return (
    <div className="{component.lower()}">
      <h2>{component}</h2>
      <p>This is the {component} component.</p>
    </div>
  );
}};

export default {component};"""
    
    async def _generate_static_website(self, architecture: Dict[str, Any], result: Dict[str, Any]):
        """Generate static HTML/CSS/JS website"""
        app_type = architecture.get('app_type', 'business')
        
        # Generate index.html
        index_content = self._generate_static_index_html(architecture)
        file_result = await self.code_generator.create_file('index.html', index_content, 'Main HTML page')
        if file_result.get('success'):
            result['files'].append(file_result)
        
        # Generate additional pages
        for page in architecture.get('pages', []):
            if page != 'home':
                page_content = self._generate_static_page_html(page, architecture)
                file_result = await self.code_generator.create_file(f'{page}.html', page_content, f'{page.title()} page')
                if file_result.get('success'):
                    result['files'].append(file_result)
                    result['pages_created'] += 1
        
        # Generate CSS
        css_content = self._generate_static_css(architecture)
        file_result = await self.code_generator.create_file('styles/main.css', css_content, 'Main stylesheet')
        if file_result.get('success'):
            result['files'].append(file_result)
        
        # Generate JavaScript
        js_content = self._generate_static_javascript(architecture)
        file_result = await self.code_generator.create_file('scripts/main.js', js_content, 'Main JavaScript file')
        if file_result.get('success'):
            result['files'].append(file_result)
        
        result['components_created'] = len(architecture.get('components', []))
    
    def _generate_static_index_html(self, architecture: Dict[str, Any]) -> str:
        """Generate static HTML index page"""
        app_type = architecture.get('app_type', 'business')
        pages = architecture.get('pages', [])
        
        nav_links = []
        for page in pages:
            if page == 'home':
                nav_links.append('          <a href="index.html">Home</a>')
            else:
                nav_links.append(f'          <a href="{page}.html">{page.replace("-", " ").title()}</a>')
        
        if app_type == 'restaurant':
            content = '''    <!-- Hero Section -->
    <section class="hero">
      <div class="hero-content">
        <h1>Welcome to Our Restaurant</h1>
        <p>Experience exceptional dining with fresh, locally sourced ingredients</p>
        <a href="menu.html" class="cta-button">View Menu</a>
      </div>
    </section>

    <!-- Features Section -->
    <section class="features">
      <div class="container">
        <h2>Why Dine With Us</h2>
        <div class="features-grid">
          <div class="feature">
            <h3>Fresh Ingredients</h3>
            <p>We source the freshest ingredients daily</p>
          </div>
          <div class="feature">
            <h3>Expert Chefs</h3>
            <p>Our experienced chefs create culinary masterpieces</p>
          </div>
          <div class="feature">
            <h3>Cozy Atmosphere</h3>
            <p>Perfect ambiance for any occasion</p>
          </div>
        </div>
      </div>
    </section>'''
        else:
            content = '''    <!-- Hero Section -->
    <section class="hero">
      <div class="container">
        <div class="hero-content">
          <h1>Welcome to Our Business</h1>
          <p>Your trusted partner for exceptional services and solutions</p>
          <a href="services.html" class="cta-button">Learn More</a>
        </div>
      </div>
    </section>

    <!-- Services Section -->
    <section class="services">
      <div class="container">
        <h2>Our Services</h2>
        <div class="services-grid">
          <div class="service">
            <h3>Service One</h3>
            <p>Description of your first service</p>
          </div>
          <div class="service">
            <h3>Service Two</h3>
            <p>Description of your second service</p>
          </div>
          <div class="service">
            <h3>Service Three</h3>
            <p>Description of your third service</p>
          </div>
        </div>
      </div>
    </section>'''
        
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Your Business - Home</title>
    <link rel="stylesheet" href="styles/main.css">
</head>
<body>
    <!-- Navigation -->
    <nav class="navigation">
        <div class="container">
            <div class="logo">
                <h2>Your Brand</h2>
            </div>
            <div class="nav-links">
{chr(10).join(nav_links)}
            </div>
        </div>
    </nav>

    <!-- Main Content -->
    <main>
{content}
    </main>

    <!-- Footer -->
    <footer class="footer">
        <div class="container">
            <div class="footer-content">
                <div class="footer-section">
                    <h3>Your Brand</h3>
                    <p>Your business description goes here.</p>
                </div>
                <div class="footer-section">
                    <h4>Contact Info</h4>
                    <p>Email: info@yourbusiness.com</p>
                    <p>Phone: (555) 123-4567</p>
                </div>
            </div>
            <div class="footer-bottom">
                <p>&copy; 2024 Your Brand. All rights reserved.</p>
            </div>
        </div>
    </footer>

    <script src="scripts/main.js"></script>
</body>
</html>"""
    
    def _generate_static_page_html(self, page: str, architecture: Dict[str, Any]) -> str:
        """Generate HTML for additional pages"""
        app_type = architecture.get('app_type', 'business')
        pages = architecture.get('pages', [])
        
        # Generate navigation
        nav_links = []
        for nav_page in pages:
            if nav_page == 'home':
                nav_links.append('          <a href="index.html">Home</a>')
            elif nav_page == page:
                nav_links.append(f'          <a href="{nav_page}.html" class="active">{nav_page.replace("-", " ").title()}</a>')
            else:
                nav_links.append(f'          <a href="{nav_page}.html">{nav_page.replace("-", " ").title()}</a>')
        
        # Page-specific content
        if page == 'menu':
            content = '''    <section class="menu-section">
      <div class="container">
        <h1>Our Menu</h1>
        <div class="menu-categories">
          <div class="menu-category">
            <h2>Coffee & Espresso</h2>
            <div class="menu-items">
              <div class="menu-item">
                <h3>Americano <span class="price">$3.50</span></h3>
                <p>Rich espresso with hot water</p>
              </div>
              <div class="menu-item">
                <h3>Latte <span class="price">$4.50</span></h3>
                <p>Espresso with steamed milk and foam</p>
              </div>
              <div class="menu-item">
                <h3>Cappuccino <span class="price">$4.00</span></h3>
                <p>Equal parts espresso, steamed milk, and foam</p>
              </div>
            </div>
          </div>
          
          <div class="menu-category">
            <h2>Pastries & Snacks</h2>
            <div class="menu-items">
              <div class="menu-item">
                <h3>Croissant <span class="price">$2.50</span></h3>
                <p>Fresh baked buttery croissant</p>
              </div>
              <div class="menu-item">
                <h3>Muffin <span class="price">$3.00</span></h3>
                <p>Blueberry or chocolate chip</p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>'''
        elif page == 'contact':
            content = '''    <section class="contact-section">
      <div class="container">
        <h1>Contact Us</h1>
        <div class="contact-content">
          <div class="contact-info">
            <h2>Get in Touch</h2>
            <p><strong>Address:</strong> 123 Coffee Street, City, ST 12345</p>
            <p><strong>Phone:</strong> (555) 123-4567</p>
            <p><strong>Email:</strong> info@coffeeshop.com</p>
            <p><strong>Hours:</strong> Mon-Fri 6AM-8PM, Sat-Sun 7AM-9PM</p>
          </div>
          <form class="contact-form">
            <h2>Send us a message</h2>
            <input type="text" placeholder="Your Name" required>
            <input type="email" placeholder="Your Email" required>
            <textarea placeholder="Your Message" rows="5" required></textarea>
            <button type="submit">Send Message</button>
          </form>
        </div>
      </div>
    </section>'''
        elif page == 'about':
            content = '''    <section class="about-section">
      <div class="container">
        <h1>About Us</h1>
        <div class="about-content">
          <div class="about-text">
            <h2>Our Story</h2>
            <p>Welcome to our coffee shop! We've been serving the community with exceptional coffee and warm hospitality since our founding. Our passion for quality coffee drives everything we do.</p>
            
            <h2>Our Mission</h2>
            <p>To create a welcoming space where people can enjoy expertly crafted coffee, delicious food, and meaningful connections with their community.</p>
            
            <h2>What Makes Us Special</h2>
            <ul>
              <li>Locally sourced, ethically traded coffee beans</li>
              <li>Fresh baked goods made daily</li>
              <li>Experienced baristas who care about their craft</li>
              <li>A warm, inviting atmosphere</li>
            </ul>
          </div>
        </div>
      </div>
    </section>'''
        else:
            # Generic page content
            content = f'''    <section class="page-section">
      <div class="container">
        <h1>{page.replace("-", " ").title()}</h1>
        <p>Welcome to the {page.replace("-", " ")} page. This content can be customized based on your specific needs.</p>
      </div>
    </section>'''
        
        return f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{page.replace("-", " ").title()} - Your Coffee Shop</title>
    <link rel="stylesheet" href="styles/main.css">
</head>
<body>
    <!-- Navigation -->
    <nav class="navbar">
        <div class="container">
            <div class="nav-brand">
                <h2>Your Coffee Shop</h2>
            </div>
            <div class="nav-menu">
{chr(10).join(nav_links)}
            </div>
        </div>
    </nav>

    <main>
{content}
    </main>

    <!-- Footer -->
    <footer class="footer">
        <div class="container">
            <div class="footer-content">
                <div class="footer-section">
                    <h3>Your Coffee Shop</h3>
                    <p>Serving exceptional coffee and creating community connections.</p>
                </div>
                <div class="footer-section">
                    <h4>Contact Info</h4>
                    <p>Email: info@coffeeshop.com</p>
                    <p>Phone: (555) 123-4567</p>
                </div>
            </div>
            <div class="footer-bottom">
                <p>&copy; 2024 Your Coffee Shop. All rights reserved.</p>
            </div>
        </div>
    </footer>

    <script src="scripts/main.js"></script>
</body>
</html>'''
    
    def _generate_static_css(self, architecture: Dict[str, Any]) -> str:
        """Generate comprehensive CSS for static website"""
        return """/* Reset and Base Styles */
* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

body {
    font-family: 'Arial', sans-serif;
    line-height: 1.6;
    color: #333;
}

.container {
    max-width: 1200px;
    margin: 0 auto;
    padding: 0 20px;
}

/* Navigation */
.navigation {
    background: #fff;
    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    position: sticky;
    top: 0;
    z-index: 100;
}

.navigation .container {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 1rem 20px;
}

.logo h2 {
    color: #2c5aa0;
}

.nav-links {
    display: flex;
    gap: 2rem;
}

.nav-links a {
    text-decoration: none;
    color: #333;
    font-weight: 500;
    transition: color 0.3s ease;
}

.nav-links a:hover {
    color: #2c5aa0;
}

/* Hero Section */
.hero {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    text-align: center;
    padding: 6rem 0;
    min-height: 70vh;
    display: flex;
    align-items: center;
    justify-content: center;
}

.hero-content h1 {
    font-size: 3.5rem;
    margin-bottom: 1rem;
    font-weight: 700;
}

.hero-content p {
    font-size: 1.2rem;
    margin-bottom: 2rem;
    max-width: 600px;
    margin-left: auto;
    margin-right: auto;
}

.cta-button {
    display: inline-block;
    background: #ff6b6b;
    color: white;
    padding: 1rem 2rem;
    text-decoration: none;
    border-radius: 5px;
    font-weight: 600;
    transition: background 0.3s ease;
    font-size: 1.1rem;
}

.cta-button:hover {
    background: #ff5252;
    transform: translateY(-2px);
}

/* Features/Services Section */
.features, .services {
    padding: 5rem 0;
    background: #f8f9fa;
}

.features h2, .services h2 {
    text-align: center;
    margin-bottom: 3rem;
    font-size: 2.5rem;
    color: #2c5aa0;
}

.features-grid, .services-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
    gap: 2rem;
    margin-top: 2rem;
}

.feature, .service {
    background: white;
    padding: 2rem;
    border-radius: 10px;
    box-shadow: 0 5px 15px rgba(0,0,0,0.1);
    text-align: center;
    transition: transform 0.3s ease;
}

.feature:hover, .service:hover {
    transform: translateY(-5px);
}

.feature h3, .service h3 {
    color: #2c5aa0;
    margin-bottom: 1rem;
    font-size: 1.5rem;
}

.feature p, .service p {
    color: #666;
    line-height: 1.6;
}

/* Footer */
.footer {
    background: #2c3e50;
    color: white;
    padding: 3rem 0 1rem;
}

.footer-content {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
    gap: 2rem;
    margin-bottom: 2rem;
}

.footer-section h3, .footer-section h4 {
    margin-bottom: 1rem;
    color: #ecf0f1;
}

.footer-section p, .footer-section li {
    color: #bdc3c7;
    line-height: 1.6;
}

.footer-section ul {
    list-style: none;
}

.footer-section ul li {
    margin-bottom: 0.5rem;
}

.footer-section a {
    color: #bdc3c7;
    text-decoration: none;
    transition: color 0.3s ease;
}

.footer-section a:hover {
    color: #3498db;
}

.footer-bottom {
    border-top: 1px solid #34495e;
    padding-top: 1rem;
    text-align: center;
    color: #95a5a6;
}

/* Responsive Design */
@media (max-width: 768px) {
    .hero-content h1 {
        font-size: 2.5rem;
    }
    
    .nav-links {
        gap: 1rem;
    }
    
    .features-grid, .services-grid {
        grid-template-columns: 1fr;
    }
    
    .navigation .container {
        flex-direction: column;
        gap: 1rem;
    }
}

/* Menu Page Specific Styles */
.menu-category {
    margin-bottom: 3rem;
}

.menu-category h3 {
    color: #2c5aa0;
    border-bottom: 2px solid #2c5aa0;
    padding-bottom: 0.5rem;
    margin-bottom: 1.5rem;
    font-size: 1.8rem;
}

.menu-item {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    padding: 1rem;
    margin-bottom: 1rem;
    background: white;
    border-radius: 5px;
    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
}

.menu-item-info h4 {
    color: #2c5aa0;
    margin-bottom: 0.5rem;
}

.menu-item-info p {
    color: #666;
    font-size: 0.9rem;
}

.menu-item-price {
    color: #e74c3c;
    font-weight: bold;
    font-size: 1.2rem;
}

/* Contact Form Styles */
.contact-form {
    background: white;
    padding: 2rem;
    border-radius: 10px;
    box-shadow: 0 5px 15px rgba(0,0,0,0.1);
    max-width: 600px;
    margin: 0 auto;
}

.form-group {
    margin-bottom: 1.5rem;
}

.form-group label {
    display: block;
    margin-bottom: 0.5rem;
    color: #2c5aa0;
    font-weight: 500;
}

.form-group input,
.form-group textarea {
    width: 100%;
    padding: 0.75rem;
    border: 2px solid #e1e8ed;
    border-radius: 5px;
    font-size: 1rem;
    transition: border-color 0.3s ease;
}

.form-group input:focus,
.form-group textarea:focus {
    outline: none;
    border-color: #2c5aa0;
}

.form-group textarea {
    height: 120px;
    resize: vertical;
}

.submit-button {
    background: #2c5aa0;
    color: white;
    padding: 0.75rem 2rem;
    border: none;
    border-radius: 5px;
    font-size: 1rem;
    cursor: pointer;
    transition: background 0.3s ease;
}

.submit-button:hover {
    background: #1e3f73;
}"""
    
    def _generate_static_javascript(self, architecture: Dict[str, Any]) -> str:
        """Generate JavaScript for static website"""
        features = architecture.get('features', [])
        
        js_content = """// Main JavaScript functionality
document.addEventListener('DOMContentLoaded', function() {
    // Mobile menu toggle
    const nav = document.querySelector('.navigation');
    const navLinks = document.querySelector('.nav-links');
    
    // Smooth scrolling for anchor links
    const links = document.querySelectorAll('a[href^="#"]');
    links.forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });
    
    // Form submission handling
    const contactForm = document.querySelector('.contact-form');
    if (contactForm) {
        contactForm.addEventListener('submit', function(e) {
            e.preventDefault();
            handleFormSubmission(this);
        });
    }
    
    // Animation on scroll
    const observeElements = document.querySelectorAll('.feature, .service');
    if (observeElements.length > 0) {
        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    entry.target.style.opacity = '1';
                    entry.target.style.transform = 'translateY(0)';
                }
            });
        });
        
        observeElements.forEach(el => {
            el.style.opacity = '0';
            el.style.transform = 'translateY(20px)';
            el.style.transition = 'opacity 0.6s ease, transform 0.6s ease';
            observer.observe(el);
        });
    }
});

function handleFormSubmission(form) {
    const formData = new FormData(form);
    const data = Object.fromEntries(formData);
    
    // Here you would typically send the data to a server
    console.log('Form submitted with data:', data);
    
    // Show success message
    showMessage('Thank you for your message! We\\'ll get back to you soon.', 'success');
    form.reset();
}

function showMessage(message, type) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${type}`;
    messageDiv.textContent = message;
    messageDiv.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        background: ${type === 'success' ? '#2ecc71' : '#e74c3c'};
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 5px;
        box-shadow: 0 3px 10px rgba(0,0,0,0.2);
        z-index: 1000;
        animation: slideIn 0.3s ease;
    `;
    
    document.body.appendChild(messageDiv);
    
    setTimeout(() => {
        messageDiv.style.animation = 'slideOut 0.3s ease';
        setTimeout(() => messageDiv.remove(), 300);
    }, 3000);
}

// Add CSS for animations
const style = document.createElement('style');
style.textContent = `
    @keyframes slideIn {
        from { transform: translateX(100%); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    @keyframes slideOut {
        from { transform: translateX(0); opacity: 1; }
        to { transform: translateX(100%); opacity: 0; }
    }
`;
document.head.appendChild(style);"""
        
        # Add shopping cart functionality if needed
        if 'shopping_cart' in features:
            js_content += """

// Shopping Cart Functionality
class ShoppingCart {
    constructor() {
        this.items = JSON.parse(localStorage.getItem('cart')) || [];
        this.updateCartDisplay();
    }
    
    addItem(product) {
        const existingItem = this.items.find(item => item.id === product.id);
        if (existingItem) {
            existingItem.quantity += 1;
        } else {
            this.items.push({...product, quantity: 1});
        }
        this.saveCart();
        this.updateCartDisplay();
    }
    
    removeItem(productId) {
        this.items = this.items.filter(item => item.id !== productId);
        this.saveCart();
        this.updateCartDisplay();
    }
    
    updateQuantity(productId, quantity) {
        const item = this.items.find(item => item.id === productId);
        if (item) {
            item.quantity = Math.max(0, quantity);
            if (item.quantity === 0) {
                this.removeItem(productId);
            } else {
                this.saveCart();
                this.updateCartDisplay();
            }
        }
    }
    
    getTotal() {
        return this.items.reduce((total, item) => total + (item.price * item.quantity), 0);
    }
    
    getItemCount() {
        return this.items.reduce((count, item) => count + item.quantity, 0);
    }
    
    saveCart() {
        localStorage.setItem('cart', JSON.stringify(this.items));
    }
    
    updateCartDisplay() {
        const cartCount = document.querySelector('.cart-count');
        if (cartCount) {
            cartCount.textContent = this.getItemCount();
        }
    }
    
    clear() {
        this.items = [];
        this.saveCart();
        this.updateCartDisplay();
    }
}

// Initialize cart
const cart = new ShoppingCart();"""
        
        return js_content
    
    async def _create_deployment_config(self, architecture: Dict[str, Any]) -> Dict[str, Any]:
        """Create deployment configuration files"""
        deployment_config = {
            'created_files': [],
            'instructions': []
        }
        
        tech_stack = architecture.get('tech_stack', 'html_css_js')
        
        try:
            if tech_stack == 'react_node_mongodb':
                # Create Docker configuration
                dockerfile_content = """# Frontend Dockerfile
FROM node:18-alpine as build
WORKDIR /app
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

FROM nginx:alpine
COPY --from=build /app/build /usr/share/nginx/html
COPY nginx.conf /etc/nginx/nginx.conf
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]"""
                
                await self.code_generator.create_file('Dockerfile', dockerfile_content, 'Docker configuration')
                
                # Create docker-compose.yml
                compose_content = """version: '3.8'
services:
  frontend:
    build: .
    ports:
      - "3000:80"
    depends_on:
      - backend
      
  backend:
    build: ./backend
    ports:
      - "5000:5000"
    environment:
      - NODE_ENV=production
      - MONGODB_URI=mongodb://mongo:27017/app
    depends_on:
      - mongo
      
  mongo:
    image: mongo:5.0
    ports:
      - "27017:27017"
    volumes:
      - mongo_data:/data/db

volumes:
  mongo_data:"""
                
                await self.code_generator.create_file('docker-compose.yml', compose_content, 'Docker Compose configuration')
                
                deployment_config['created_files'].extend(['Dockerfile', 'docker-compose.yml'])
                deployment_config['instructions'].extend([
                    'Run: docker-compose up --build',
                    'Access frontend at: http://localhost:3000',
                    'Backend API at: http://localhost:5000'
                ])
                
            elif tech_stack == 'html_css_js':
                # Create simple deployment instructions
                deployment_config['instructions'].extend([
                    'Upload all files to your web server',
                    'Ensure index.html is in the root directory',
                    'Configure your web server to serve static files',
                    'For GitHub Pages: commit and push to gh-pages branch'
                ])
                
            elif tech_stack == 'flutter':
                # Create build instructions for Flutter
                await self.code_generator.create_file(
                    'build_instructions.md',
                    """# Flutter Build Instructions

## Development
```bash
flutter pub get
flutter run
```

## Production Build

### Android APK
```bash
flutter build apk --release
```

### iOS (macOS required)
```bash
flutter build ios --release
```

### Web
```bash
flutter build web
```

## Deployment
- Android: Upload APK to Google Play Store
- iOS: Upload to App Store Connect
- Web: Deploy build/web folder to web hosting
""",
                    'Flutter build instructions'
                )
                
                deployment_config['created_files'].append('build_instructions.md')
                deployment_config['instructions'].extend([
                    'Run: flutter pub get',
                    'For development: flutter run',
                    'For production: flutter build apk --release'
                ])
            
            return deployment_config
            
        except Exception as e:
            print(f"[DEBUG] Error creating deployment config: {e}")
            return deployment_config
    
    async def _generate_documentation(self, architecture: Dict[str, Any], generation_result: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive documentation"""
        documentation = {
            'readme_created': False,
            'api_docs_created': False,
            'user_guide_created': False
        }
        
        try:
            # Generate README.md
            readme_content = self._generate_readme_content(architecture, generation_result)
            await self.code_generator.create_file('README.md', readme_content, 'Project documentation')
            documentation['readme_created'] = True
            
            # Generate API documentation if it's a backend project
            if architecture.get('api_endpoints'):
                api_docs = self._generate_api_documentation(architecture)
                await self.code_generator.create_file('API.md', api_docs, 'API documentation')
                documentation['api_docs_created'] = True
            
            return documentation
            
        except Exception as e:
            print(f"[DEBUG] Error generating documentation: {e}")
            return documentation
    
    def _generate_readme_content(self, architecture: Dict[str, Any], generation_result: Dict[str, Any]) -> str:
        """Generate comprehensive README content"""
        app_type = architecture.get('app_type', 'application')
        tech_stack = architecture.get('tech_stack', 'html_css_js')
        features = architecture.get('features', [])
        
        # Technology stack description
        tech_descriptions = {
            'react_node_mongodb': 'React.js frontend with Node.js backend and MongoDB database',
            'html_css_js_backend': 'Static HTML/CSS/JavaScript frontend with backend API',
            'html_css_js': 'Static HTML/CSS/JavaScript website',
            'flutter': 'Cross-platform Flutter mobile application',
            'fastapi_python': 'FastAPI Python backend service'
        }
        
        tech_desc = tech_descriptions.get(tech_stack, 'Custom technology stack')
        
        return f"""# {app_type.replace('_', ' ').title()} Application

🚀 **Built with ABOV3 Genesis - From Idea to Built Reality**

## Description

This is a complete {app_type.replace('_', ' ')} application built with {tech_desc}.

## Features

{chr(10).join([f'- {feature.replace("_", " ").title()}' for feature in features]) if features else '- Modern, responsive design\n- User-friendly interface\n- Cross-browser compatibility'}

## Technology Stack

- **Frontend**: {self._get_frontend_tech(tech_stack)}
- **Backend**: {self._get_backend_tech(tech_stack)}
- **Database**: {self._get_database_tech(tech_stack)}

## Project Structure

```
{self._get_project_structure(architecture)}
```

## Installation & Setup

### Prerequisites
{self._get_prerequisites(tech_stack)}

### Installation Steps

1. Clone the repository:
   ```bash
   git clone <your-repo-url>
   cd {app_type}-app
   ```

{self._get_installation_steps(tech_stack)}

## Usage

{self._get_usage_instructions(tech_stack)}

## Development

### Running in Development Mode
{self._get_dev_instructions(tech_stack)}

### Building for Production
{self._get_build_instructions(tech_stack)}

## API Endpoints

{self._format_api_endpoints(architecture.get('api_endpoints', [])) if architecture.get('api_endpoints') else 'This application does not expose API endpoints.'}

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For support and questions, please open an issue in the repository.

---

**Generated with ABOV3 Genesis** - Your AI-powered development assistant that transforms ideas into production-ready applications.
"""
    
    def _get_frontend_tech(self, tech_stack: str) -> str:
        tech_map = {
            'react_node_mongodb': 'React.js 18, React Router, Axios',
            'html_css_js_backend': 'HTML5, CSS3, Vanilla JavaScript',
            'html_css_js': 'HTML5, CSS3, Vanilla JavaScript',
            'flutter': 'Flutter SDK, Dart',
            'fastapi_python': 'None (API only)'
        }
        return tech_map.get(tech_stack, 'Custom frontend')
    
    def _get_backend_tech(self, tech_stack: str) -> str:
        tech_map = {
            'react_node_mongodb': 'Node.js, Express.js, JWT Authentication',
            'html_css_js_backend': 'Node.js, Express.js',
            'html_css_js': 'None (Static site)',
            'flutter': 'RESTful API integration',
            'fastapi_python': 'FastAPI, Python 3.9+'
        }
        return tech_map.get(tech_stack, 'Custom backend')
    
    def _get_database_tech(self, tech_stack: str) -> str:
        tech_map = {
            'react_node_mongodb': 'MongoDB with Mongoose ODM',
            'html_css_js_backend': 'MongoDB or PostgreSQL',
            'html_css_js': 'None (Static site)',
            'flutter': 'SQLite (local) + Remote API',
            'fastapi_python': 'PostgreSQL with SQLAlchemy ORM'
        }
        return tech_map.get(tech_stack, 'Custom database')
    
    def _get_prerequisites(self, tech_stack: str) -> str:
        prereq_map = {
            'react_node_mongodb': '- Node.js 16+ and npm\n- MongoDB 5.0+\n- Git',
            'html_css_js_backend': '- Node.js 16+ and npm\n- Git',
            'html_css_js': '- Web browser\n- Text editor\n- Web server (optional)',
            'flutter': '- Flutter SDK 3.0+\n- Dart SDK\n- Android Studio / Xcode\n- Git',
            'fastapi_python': '- Python 3.9+\n- pip package manager\n- Git'
        }
        return prereq_map.get(tech_stack, '- Basic development environment')
    
    def _get_installation_steps(self, tech_stack: str) -> str:
        if tech_stack == 'react_node_mongodb':
            return """2. Install frontend dependencies:
   ```bash
   cd frontend
   npm install
   ```

3. Install backend dependencies:
   ```bash
   cd backend
   npm install
   ```

4. Set up environment variables:
   ```bash
   cd backend
   echo "MONGODB_URI=mongodb://localhost:27017/app" > .env
   echo "JWT_SECRET=your-secret-key-here" >> .env
   echo "PORT=5000" >> .env
   ```

5. Start MongoDB service (make sure MongoDB is installed and running)
   ```bash
   # On Windows with MongoDB installed
   net start MongoDB
   
   # On macOS with Homebrew
   brew services start mongodb-community
   
   # On Linux
   sudo systemctl start mongod
   ```"""
        
        elif tech_stack == 'flutter':
            return """2. Get Flutter dependencies:
   ```bash
   flutter pub get
   ```

3. Check Flutter doctor:
   ```bash
   flutter doctor
   ```"""
        
        elif tech_stack == 'fastapi_python':
            return """2. Create virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\\Scripts\\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```"""
        
        else:
            return """2. Open the project in your preferred code editor

3. If using a local server, start it in the project directory"""
    
    def _get_usage_instructions(self, tech_stack: str) -> str:
        if tech_stack == 'react_node_mongodb':
            return """1. Start the backend server:
   ```bash
   cd backend && npm start
   ```

2. Start the frontend development server:
   ```bash
   cd frontend && npm start
   ```

3. Open your browser and navigate to `http://localhost:3000`"""
        
        elif tech_stack == 'flutter':
            return """1. Run the application:
   ```bash
   flutter run
   ```

2. The app will launch on your connected device or emulator"""
        
        elif tech_stack == 'fastapi_python':
            return """1. Start the API server:
   ```bash
   uvicorn main:app --reload
   ```

2. Access the API documentation at `http://localhost:8000/docs`"""
        
        else:
            return """1. Open `index.html` in your web browser
2. For local development, use a local server like Python's built-in server:
   ```bash
   python -m http.server 8000
   ```
3. Navigate to `http://localhost:8000`"""
    
    def _get_dev_instructions(self, tech_stack: str) -> str:
        if tech_stack == 'react_node_mongodb':
            return """```bash
# Terminal 1 - Backend
cd backend
npm install
npm run dev

# Terminal 2 - Frontend  
cd frontend
npm install
npm start
```"""
        elif tech_stack == 'flutter':
            return """```bash
flutter run --debug
```"""
        elif tech_stack == 'fastapi_python':
            return """```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```"""
        else:
            return """```bash
# Use any local server
python -m http.server 8000
# Or use Live Server extension in VS Code
```"""
    
    def _get_build_instructions(self, tech_stack: str) -> str:
        if tech_stack == 'react_node_mongodb':
            return """```bash
# Build frontend for production
cd frontend
npm run build

# Build backend for production
cd ../backend
npm run build
```"""
        elif tech_stack == 'flutter':
            return """```bash
# Android APK
flutter build apk --release

# iOS (macOS required)
flutter build ios --release

# Web
flutter build web
```"""
        elif tech_stack == 'fastapi_python':
            return """```bash
# Install production dependencies
pip install -r requirements.txt

# Run with production server
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker
```"""
        else:
            return """```bash
# No build step required for static sites
# Just upload all files to your web server
```"""
    
    def _get_project_structure(self, architecture: Dict[str, Any]) -> str:
        tech_stack = architecture.get('tech_stack', 'html_css_js')
        
        if tech_stack == 'react_node_mongodb':
            return """├── frontend/
│   ├── public/
│   ├── src/
│   │   ├── components/
│   │   ├── pages/
│   │   ├── contexts/
│   │   └── services/
│   └── package.json
├── backend/
│   ├── routes/
│   ├── models/
│   ├── middleware/
│   └── server.js
├── docker-compose.yml
└── README.md"""
        
        elif tech_stack == 'flutter':
            return """├── lib/
│   ├── screens/
│   ├── widgets/
│   ├── models/
│   ├── services/
│   └── main.dart
├── assets/
│   └── images/
├── pubspec.yaml
└── README.md"""
        
        else:
            return """├── index.html
├── styles/
│   └── main.css
├── scripts/
│   └── main.js
├── assets/
│   └── images/
└── README.md"""
    
    def _format_api_endpoints(self, endpoints: List[str]) -> str:
        if not endpoints:
            return 'No API endpoints defined.'
        
        formatted = []
        for endpoint in endpoints:
            formatted.append(f'- `{endpoint}`')
        
        return '\n'.join(formatted)
    
    def _get_next_steps(self, architecture: Dict[str, Any]) -> List[str]:
        """Generate next steps for the user"""
        tech_stack = architecture.get('tech_stack', 'html_css_js')
        features = architecture.get('features', [])
        
        steps = [
            "Review and customize the generated code",
            "Update branding, colors, and styling to match your needs",
            "Add your actual content and data"
        ]
        
        if 'user_auth' in features:
            steps.append("Configure authentication providers")
        
        if 'payment' in features:
            steps.append("Set up payment gateway (Stripe, PayPal, etc.)")
        
        if 'database' in features:
            steps.append("Configure production database connection")
        
        if tech_stack in ['react_node_mongodb', 'html_css_js_backend', 'fastapi_python']:
            steps.append("Deploy backend to cloud service (Heroku, AWS, etc.)")
            steps.append("Set up environment variables for production")
        
        steps.extend([
            "Test the application thoroughly",
            "Set up monitoring and analytics",
            "Deploy to production environment"
        ])
        
        return steps
    
    def _should_create_deployment_config(self, description: str, preferences: Dict[str, Any] = None) -> bool:
        """Check if user specifically requested deployment configurations"""
        # NEVER create deployment configurations unless explicitly requested with 'docker' or 'deploy'
        description_lower = description.lower()
        
        explicit_deployment_keywords = ['docker', 'deployment', 'deploy']
        
        # Only if user explicitly asks for deployment
        if any(keyword in description_lower for keyword in explicit_deployment_keywords):
            return True
        
        return False
    
    async def _generate_node_backend(self, architecture: Dict[str, Any], result: Dict[str, Any]):
        """Generate Node.js backend for React applications"""
        app_type = architecture.get('app_type', 'business')
        
        # Generate backend package.json
        backend_package = {
            "name": f"{app_type}-backend",
            "version": "1.0.0",
            "description": f"Backend API for {app_type} application",
            "main": "server.js",
            "scripts": {
                "start": "node server.js",
                "dev": "nodemon server.js",
                "test": "jest"
            },
            "dependencies": {
                "express": "^4.18.2",
                "cors": "^2.8.5",
                "helmet": "^6.0.1",
                "morgan": "^1.10.0",
                "dotenv": "^16.0.3"
            },
            "devDependencies": {
                "nodemon": "^2.0.20",
                "jest": "^29.3.1"
            }
        }
        
        # Add specific dependencies based on features
        if 'user_auth' in architecture.get('features', []):
            backend_package['dependencies'].update({
                "bcryptjs": "^2.4.3",
                "jsonwebtoken": "^9.0.0"
            })
        
        if 'database' in architecture.get('features', []) or app_type == 'ecommerce':
            backend_package['dependencies'].update({
                "mongoose": "^6.8.4"
            })
        
        if 'shopping_cart' in architecture.get('features', []):
            backend_package['dependencies'].update({
                "stripe": "^11.6.0"
            })
        
        file_result = await self.code_generator.create_file(
            'backend/package.json',
            json.dumps(backend_package, indent=2),
            'Backend package configuration'
        )
        if file_result.get('success'):
            result['files'].append(file_result)
        
        # Generate main server.js
        server_content = self._generate_express_server(architecture)
        file_result = await self.code_generator.create_file('backend/server.js', server_content, 'Express server')
        if file_result.get('success'):
            result['files'].append(file_result)
        
        # Generate routes
        for endpoint_group in ['auth', 'api']:
            route_content = self._generate_express_route(endpoint_group, architecture)
            file_result = await self.code_generator.create_file(
                f'backend/routes/{endpoint_group}.js',
                route_content,
                f'{endpoint_group.title()} routes'
            )
            if file_result.get('success'):
                result['files'].append(file_result)
        
        # Generate models if using database
        if 'database' in architecture.get('features', []) or app_type == 'ecommerce':
            models = ['User', 'Product', 'Order'] if app_type == 'ecommerce' else ['User', 'Data']
            for model in models:
                model_content = self._generate_mongoose_model(model, architecture)
                file_result = await self.code_generator.create_file(
                    f'backend/models/{model}.js',
                    model_content,
                    f'{model} model'
                )
                if file_result.get('success'):
                    result['files'].append(file_result)
    
    def _generate_express_server(self, architecture: Dict[str, Any]) -> str:
        """Generate Express.js server file"""
        app_type = architecture.get('app_type', 'business')
        features = architecture.get('features', [])
        
        imports = [
            "const express = require('express');",
            "const cors = require('cors');",
            "const helmet = require('helmet');",
            "const morgan = require('morgan');",
            "require('dotenv').config();"
        ]
        
        if 'database' in features or app_type == 'ecommerce':
            imports.append("const mongoose = require('mongoose');")
        
        middleware_setup = [
            "// Middleware",
            "app.use(helmet());",
            "app.use(cors());",
            "app.use(morgan('combined'));",
            "app.use(express.json());",
            "app.use(express.urlencoded({ extended: true }));"
        ]
        
        routes_setup = [
            "// Routes",
            "app.use('/api/auth', require('./routes/auth'));",
            "app.use('/api', require('./routes/api'));"
        ]
        
        db_setup = ""
        if 'database' in features or app_type == 'ecommerce':
            db_setup = """
// Database connection
mongoose.connect(process.env.MONGODB_URI || 'mongodb://localhost:27017/app', {
  useNewUrlParser: true,
  useUnifiedTopology: true,
});

mongoose.connection.on('connected', () => {
  console.log('[OK] Connected to MongoDB');
});

mongoose.connection.on('error', (err) => {
  console.error('[ERROR] MongoDB connection error:', err);
});"""
        
        return f"""const express = require('express');
{chr(10).join(imports)}

const app = express();
const PORT = process.env.PORT || 5000;

{chr(10).join(middleware_setup)}
{db_setup}

{chr(10).join(routes_setup)}

// Health check endpoint
app.get('/health', (req, res) => {{
  res.json({{ status: 'OK', message: 'Server is running!' }});
}});

// Error handling middleware
app.use((err, req, res, next) => {{
  console.error(err.stack);
  res.status(500).json({{ message: 'Something went wrong!' }});
}});

// 404 handler
app.use((req, res) => {{
  res.status(404).json({{ message: 'Route not found' }});
}});

app.listen(PORT, () => {{
  console.log(`🚀 Server running on port ${{PORT}}`);
}});"""
    
    def _generate_express_route(self, route_type: str, architecture: Dict[str, Any]) -> str:
        """Generate Express route files"""
        if route_type == 'auth':
            return """const express = require('express');
const router = express.Router();

// Register endpoint
router.post('/register', async (req, res) => {
  try {
    const { email, password, name } = req.body;
    
    // Add validation here
    if (!email || !password || !name) {
      return res.status(400).json({ message: 'All fields are required' });
    }
    
    // Create user logic here
    res.status(201).json({ 
      message: 'User created successfully',
      user: { email, name }
    });
  } catch (error) {
    res.status(500).json({ message: 'Registration failed', error: error.message });
  }
});

// Login endpoint
router.post('/login', async (req, res) => {
  try {
    const { email, password } = req.body;
    
    // Add authentication logic here
    if (!email || !password) {
      return res.status(400).json({ message: 'Email and password are required' });
    }
    
    // Authenticate user and return token
    res.json({ 
      message: 'Login successful',
      token: 'your-jwt-token-here',
      user: { email }
    });
  } catch (error) {
    res.status(500).json({ message: 'Login failed', error: error.message });
  }
});

module.exports = router;"""
        
        elif route_type == 'api':
            app_type = architecture.get('app_type', 'business')
            
            if app_type == 'ecommerce':
                return """const express = require('express');
const router = express.Router();

// Get all products
router.get('/products', async (req, res) => {
  try {
    // Fetch products from database
    const products = [
      { id: 1, name: 'Sample Product', price: 29.99, description: 'A great product' },
      { id: 2, name: 'Another Product', price: 39.99, description: 'Another great product' }
    ];
    res.json(products);
  } catch (error) {
    res.status(500).json({ message: 'Failed to fetch products' });
  }
});

// Get single product
router.get('/products/:id', async (req, res) => {
  try {
    const { id } = req.params;
    // Fetch single product from database
    const product = { id, name: 'Sample Product', price: 29.99, description: 'A great product' };
    res.json(product);
  } catch (error) {
    res.status(500).json({ message: 'Product not found' });
  }
});

// Add to cart
router.post('/cart/add', async (req, res) => {
  try {
    const { productId, quantity } = req.body;
    // Add cart logic here
    res.json({ message: 'Product added to cart', productId, quantity });
  } catch (error) {
    res.status(500).json({ message: 'Failed to add to cart' });
  }
});

// Create order
router.post('/orders', async (req, res) => {
  try {
    const { items, total } = req.body;
    // Create order logic here
    res.status(201).json({ 
      message: 'Order created successfully',
      orderId: Date.now(),
      total
    });
  } catch (error) {
    res.status(500).json({ message: 'Failed to create order' });
  }
});

module.exports = router;"""
            
            else:
                return """const express = require('express');
const router = express.Router();

// Get data endpoint
router.get('/data', async (req, res) => {
  try {
    const data = [
      { id: 1, title: 'Sample Data', content: 'This is sample content' },
      { id: 2, title: 'More Data', content: 'This is more sample content' }
    ];
    res.json(data);
  } catch (error) {
    res.status(500).json({ message: 'Failed to fetch data' });
  }
});

// Create data endpoint
router.post('/data', async (req, res) => {
  try {
    const { title, content } = req.body;
    const newData = { id: Date.now(), title, content };
    res.status(201).json(newData);
  } catch (error) {
    res.status(500).json({ message: 'Failed to create data' });
  }
});

module.exports = router;"""
    
    def _generate_mongoose_model(self, model_name: str, architecture: Dict[str, Any]) -> str:
        """Generate Mongoose model files"""
        if model_name == 'User':
            return """const mongoose = require('mongoose');

const userSchema = new mongoose.Schema({
  email: {
    type: String,
    required: true,
    unique: true,
    lowercase: true
  },
  password: {
    type: String,
    required: true,
    minlength: 6
  },
  name: {
    type: String,
    required: true
  },
  role: {
    type: String,
    enum: ['user', 'admin'],
    default: 'user'
  },
  createdAt: {
    type: Date,
    default: Date.now
  }
});

module.exports = mongoose.model('User', userSchema);"""
        
        elif model_name == 'Product':
            return """const mongoose = require('mongoose');

const productSchema = new mongoose.Schema({
  name: {
    type: String,
    required: true
  },
  description: {
    type: String,
    required: true
  },
  price: {
    type: Number,
    required: true,
    min: 0
  },
  category: {
    type: String,
    required: true
  },
  image: {
    type: String,
    default: ''
  },
  stock: {
    type: Number,
    default: 0,
    min: 0
  },
  featured: {
    type: Boolean,
    default: false
  },
  createdAt: {
    type: Date,
    default: Date.now
  }
});

module.exports = mongoose.model('Product', productSchema);"""
        
        elif model_name == 'Order':
            return """const mongoose = require('mongoose');

const orderSchema = new mongoose.Schema({
  user: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'User',
    required: true
  },
  items: [{
    product: {
      type: mongoose.Schema.Types.ObjectId,
      ref: 'Product',
      required: true
    },
    quantity: {
      type: Number,
      required: true,
      min: 1
    },
    price: {
      type: Number,
      required: true
    }
  }],
  total: {
    type: Number,
    required: true
  },
  status: {
    type: String,
    enum: ['pending', 'processing', 'shipped', 'delivered', 'cancelled'],
    default: 'pending'
  },
  shippingAddress: {
    street: String,
    city: String,
    state: String,
    zipCode: String,
    country: String
  },
  createdAt: {
    type: Date,
    default: Date.now
  }
});

module.exports = mongoose.model('Order', orderSchema);"""
        
        else:
            return f"""const mongoose = require('mongoose');

const {model_name.lower()}Schema = new mongoose.Schema({{
  title: {{
    type: String,
    required: true
  }},
  content: {{
    type: String,
    required: true
  }},
  createdAt: {{
    type: Date,
    default: Date.now
  }}
}});

module.exports = mongoose.model('{model_name}', {model_name.lower()}Schema);"""
    
    async def _generate_react_styles(self, architecture: Dict[str, Any]):
        """Generate CSS styles for React application"""
        global_css = """/* Global Styles */
* {
  margin: 0;
  padding: 0;
  box-sizing: border-box;
}

body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Oxygen',
    'Ubuntu', 'Cantarell', 'Fira Sans', 'Droid Sans', 'Helvetica Neue',
    sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  line-height: 1.6;
  color: #333;
}

.container {
  max-width: 1200px;
  margin: 0 auto;
  padding: 0 20px;
}

/* Navigation Styles */
.navigation {
  background: #fff;
  box-shadow: 0 2px 4px rgba(0,0,0,0.1);
  position: sticky;
  top: 0;
  z-index: 100;
}

.navigation .container {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 1rem 20px;
}

.logo {
  font-size: 1.5rem;
  font-weight: bold;
  color: #2c5aa0;
}

.nav-links {
  display: flex;
  gap: 2rem;
}

.nav-link {
  text-decoration: none;
  color: #333;
  font-weight: 500;
  transition: color 0.3s ease;
}

.nav-link:hover {
  color: #2c5aa0;
}

/* Button Styles */
.btn {
  display: inline-block;
  padding: 0.75rem 1.5rem;
  background: #2c5aa0;
  color: white;
  text-decoration: none;
  border-radius: 4px;
  border: none;
  cursor: pointer;
  font-size: 1rem;
  transition: background 0.3s ease;
}

.btn:hover {
  background: #1e3f73;
}

.btn-secondary {
  background: #6c757d;
}

.btn-secondary:hover {
  background: #545b62;
}

/* Card Styles */
.card {
  background: white;
  border-radius: 8px;
  padding: 1.5rem;
  box-shadow: 0 2px 10px rgba(0,0,0,0.1);
  margin-bottom: 1rem;
}

/* Grid Layouts */
.grid {
  display: grid;
  gap: 1.5rem;
}

.grid-2 {
  grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
}

.grid-3 {
  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
}

.grid-4 {
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
}

/* Form Styles */
.form-group {
  margin-bottom: 1rem;
}

.form-group label {
  display: block;
  margin-bottom: 0.5rem;
  font-weight: 500;
  color: #333;
}

.form-group input,
.form-group select,
.form-group textarea {
  width: 100%;
  padding: 0.75rem;
  border: 1px solid #ddd;
  border-radius: 4px;
  font-size: 1rem;
}

.form-group input:focus,
.form-group select:focus,
.form-group textarea:focus {
  outline: none;
  border-color: #2c5aa0;
  box-shadow: 0 0 0 2px rgba(44, 90, 160, 0.2);
}

/* Utility Classes */
.text-center { text-align: center; }
.text-left { text-align: left; }
.text-right { text-align: right; }

.mt-1 { margin-top: 0.25rem; }
.mt-2 { margin-top: 0.5rem; }
.mt-3 { margin-top: 1rem; }
.mt-4 { margin-top: 1.5rem; }
.mt-5 { margin-top: 3rem; }

.mb-1 { margin-bottom: 0.25rem; }
.mb-2 { margin-bottom: 0.5rem; }
.mb-3 { margin-bottom: 1rem; }
.mb-4 { margin-bottom: 1.5rem; }
.mb-5 { margin-bottom: 3rem; }

/* Responsive Design */
@media (max-width: 768px) {
  .navigation .container {
    flex-direction: column;
    gap: 1rem;
  }
  
  .nav-links {
    gap: 1rem;
  }
  
  .grid-2,
  .grid-3,
  .grid-4 {
    grid-template-columns: 1fr;
  }
}"""
        
        await self.code_generator.create_file(
            'frontend/src/styles/global.css',
            global_css,
            'Global CSS styles'
        )
    
    async def _setup_development_environment(self, architecture: Dict[str, Any]) -> Dict[str, Any]:
        """Setup the complete development environment with dependency installation"""
        setup_result = {
            'dependencies_installed': False,
            'environment_configured': False,
            'database_setup': False,
            'dev_servers_ready': False,
            'installation_logs': [],
            'setup_commands': [],
            'errors': []
        }
        
        try:
            tech_stack = architecture.get('tech_stack', 'html_css_js')
            
            print(f"[DEBUG] Setting up development environment for {tech_stack}")
            
            if tech_stack == 'react_node_mongodb':
                await self._setup_react_node_environment(architecture, setup_result)
            elif tech_stack == 'html_css_js_backend':
                await self._setup_html_backend_environment(architecture, setup_result)
            elif tech_stack == 'flutter':
                await self._setup_flutter_environment(architecture, setup_result)
            elif tech_stack == 'fastapi_python':
                await self._setup_python_environment(architecture, setup_result)
            else:
                # Static website - no dependencies to install
                setup_result['dependencies_installed'] = True
                setup_result['dev_servers_ready'] = True
                setup_result['installation_logs'].append("Static website - no dependencies required")
            
            return setup_result
            
        except Exception as e:
            setup_result['errors'].append(f"Environment setup failed: {str(e)}")
            print(f"[DEBUG] Environment setup error: {e}")
            return setup_result
    
    async def _setup_react_node_environment(self, architecture: Dict[str, Any], setup_result: Dict[str, Any]):
        """Setup React + Node.js development environment"""
        import asyncio
        import subprocess
        import os
        
        try:
            project_path = self.project_path
            
            # Check if Node.js is available
            try:
                node_version = await self._run_command(['node', '--version'])
                npm_version = await self._run_command(['npm', '--version'])
                setup_result['installation_logs'].append(f"[OK] Node.js: {node_version.strip()}")
                setup_result['installation_logs'].append(f"[OK] npm: {npm_version.strip()}")
            except Exception:
                setup_result['errors'].append("[ERROR] Node.js not found. Please install Node.js first.")
                return
            
            # Install frontend dependencies
            frontend_path = project_path / 'frontend'
            if frontend_path.exists():
                setup_result['installation_logs'].append("[INSTALL] Installing frontend dependencies...")
                try:
                    # Check if package.json exists
                    package_json = frontend_path / 'package.json'
                    if package_json.exists():
                        install_output = await self._run_command(['npm', 'install'], cwd=str(frontend_path))
                        setup_result['installation_logs'].append("[OK] Frontend dependencies installed")
                        
                        # Create React App if using Create React App
                        if not (frontend_path / 'src').exists():
                            setup_result['installation_logs'].append("[INIT] Initializing React app structure...")
                            await self._run_command(['npx', 'create-react-app', '.'], cwd=str(frontend_path))
                        
                except Exception as e:
                    setup_result['errors'].append(f"[ERROR] Frontend installation failed: {str(e)}")
            
            # Install backend dependencies
            backend_path = project_path / 'backend'
            if backend_path.exists():
                setup_result['installation_logs'].append("[INSTALL] Installing backend dependencies...")
                try:
                    package_json = backend_path / 'package.json'
                    if package_json.exists():
                        install_output = await self._run_command(['npm', 'install'], cwd=str(backend_path))
                        setup_result['installation_logs'].append("[OK] Backend dependencies installed")
                except Exception as e:
                    setup_result['errors'].append(f"[ERROR] Backend installation failed: {str(e)}")
            
            # Setup environment variables
            await self._setup_environment_variables(architecture, setup_result)
            
            # Check MongoDB availability
            await self._setup_mongodb_environment(architecture, setup_result)
            
            if not setup_result['errors']:
                setup_result['dependencies_installed'] = True
                setup_result['dev_servers_ready'] = True
                setup_result['setup_commands'] = [
                    "cd backend && npm run dev",
                    "cd frontend && npm start"
                ]
            
        except Exception as e:
            setup_result['errors'].append(f"React/Node setup failed: {str(e)}")
    
    async def _setup_flutter_environment(self, architecture: Dict[str, Any], setup_result: Dict[str, Any]):
        """Setup Flutter development environment"""
        try:
            # Check Flutter installation
            try:
                flutter_version = await self._run_command(['flutter', '--version'])
                setup_result['installation_logs'].append("[OK] Flutter installed")
                
                # Run flutter doctor
                doctor_output = await self._run_command(['flutter', 'doctor'])
                setup_result['installation_logs'].append("[CHECK] Flutter doctor completed")
                
                # Get dependencies
                pub_output = await self._run_command(['flutter', 'pub', 'get'], cwd=str(self.project_path))
                setup_result['installation_logs'].append("[OK] Flutter dependencies installed")
                
                setup_result['dependencies_installed'] = True
                setup_result['dev_servers_ready'] = True
                setup_result['setup_commands'] = ["flutter run"]
                
            except Exception:
                setup_result['errors'].append("[ERROR] Flutter not found. Please install Flutter SDK first.")
                
        except Exception as e:
            setup_result['errors'].append(f"Flutter setup failed: {str(e)}")
    
    async def _setup_python_environment(self, architecture: Dict[str, Any], setup_result: Dict[str, Any]):
        """Setup Python/FastAPI development environment"""
        try:
            # Check Python installation
            try:
                python_version = await self._run_command(['python', '--version'])
                setup_result['installation_logs'].append(f"[OK] Python: {python_version.strip()}")
            except Exception:
                try:
                    python_version = await self._run_command(['python3', '--version'])
                    setup_result['installation_logs'].append(f"[OK] Python3: {python_version.strip()}")
                except Exception:
                    setup_result['errors'].append("[ERROR] Python not found. Please install Python first.")
                    return
            
            # Create virtual environment
            venv_path = self.project_path / 'venv'
            if not venv_path.exists():
                setup_result['installation_logs'].append("[SETUP] Creating virtual environment...")
                await self._run_command(['python', '-m', 'venv', 'venv'], cwd=str(self.project_path))
                setup_result['installation_logs'].append("[OK] Virtual environment created")
            
            # Install requirements
            requirements_file = self.project_path / 'requirements.txt'
            if requirements_file.exists():
                setup_result['installation_logs'].append("[INSTALL] Installing Python dependencies...")
                pip_cmd = str(venv_path / 'Scripts' / 'pip') if os.name == 'nt' else str(venv_path / 'bin' / 'pip')
                await self._run_command([pip_cmd, 'install', '-r', 'requirements.txt'], cwd=str(self.project_path))
                setup_result['installation_logs'].append("[OK] Python dependencies installed")
            
            setup_result['dependencies_installed'] = True
            setup_result['dev_servers_ready'] = True
            setup_result['setup_commands'] = ["uvicorn main:app --reload"]
            
        except Exception as e:
            setup_result['errors'].append(f"Python setup failed: {str(e)}")
    
    async def _setup_html_backend_environment(self, architecture: Dict[str, Any], setup_result: Dict[str, Any]):
        """Setup HTML + Backend environment"""
        # Similar to React setup but simpler
        try:
            backend_path = self.project_path / 'backend'
            if backend_path.exists():
                setup_result['installation_logs'].append("[INSTALL] Installing backend dependencies...")
                package_json = backend_path / 'package.json'
                if package_json.exists():
                    await self._run_command(['npm', 'install'], cwd=str(backend_path))
                    setup_result['installation_logs'].append("[OK] Backend dependencies installed")
            
            setup_result['dependencies_installed'] = True
            setup_result['dev_servers_ready'] = True
            setup_result['setup_commands'] = [
                "cd backend && npm run dev",
                "python -m http.server 8000  # For frontend"
            ]
            
        except Exception as e:
            setup_result['errors'].append(f"HTML/Backend setup failed: {str(e)}")
    
    async def _setup_environment_variables(self, architecture: Dict[str, Any], setup_result: Dict[str, Any]):
        """Setup environment variables for the application"""
        try:
            backend_path = self.project_path / 'backend'
            if backend_path.exists():
                env_file = backend_path / '.env'
                
                # Create .env file with default values
                env_content = []
                env_content.append("# Database Configuration")
                env_content.append("MONGODB_URI=mongodb://localhost:27017/app")
                env_content.append("")
                env_content.append("# JWT Configuration")
                env_content.append("JWT_SECRET=your-super-secret-jwt-key-change-this-in-production")
                env_content.append("")
                env_content.append("# Server Configuration")
                env_content.append("PORT=5000")
                env_content.append("NODE_ENV=development")
                
                # Add feature-specific environment variables
                features = architecture.get('features', [])
                if 'shopping_cart' in features or 'payment' in features:
                    env_content.append("")
                    env_content.append("# Payment Configuration (Stripe)")
                    env_content.append("STRIPE_SECRET_KEY=sk_test_your_stripe_secret_key_here")
                    env_content.append("STRIPE_PUBLISHABLE_KEY=pk_test_your_stripe_publishable_key_here")
                
                if 'email' in features or 'notifications' in features:
                    env_content.append("")
                    env_content.append("# Email Configuration")
                    env_content.append("SMTP_HOST=smtp.gmail.com")
                    env_content.append("SMTP_PORT=587")
                    env_content.append("SMTP_USER=your-email@gmail.com")
                    env_content.append("SMTP_PASS=your-app-password")
                
                # Write .env file
                with open(env_file, 'w') as f:
                    f.write('\n'.join(env_content))
                
                setup_result['installation_logs'].append("[OK] Environment variables configured (.env file created)")
                setup_result['environment_configured'] = True
                
        except Exception as e:
            setup_result['errors'].append(f"Environment variables setup failed: {str(e)}")
    
    async def _setup_mongodb_environment(self, architecture: Dict[str, Any], setup_result: Dict[str, Any]):
        """Check and setup MongoDB if needed"""
        try:
            features = architecture.get('features', [])
            app_type = architecture.get('app_type', '')
            
            # Check if MongoDB is needed
            if 'database' in features or app_type in ['ecommerce', 'restaurant']:
                # Try to connect to MongoDB
                try:
                    # Check if MongoDB is running
                    mongo_check = await self._run_command(['mongosh', '--eval', 'db.runCommand("ping").ok'], timeout=5)
                    if '1' in mongo_check:
                        setup_result['installation_logs'].append("[OK] MongoDB is running and accessible")
                        setup_result['database_setup'] = True
                    else:
                        setup_result['installation_logs'].append("[WARN] MongoDB found but may not be running")
                        setup_result['installation_logs'].append("[INFO] Start MongoDB with: mongod or brew services start mongodb-community")
                except Exception:
                    # Try alternative MongoDB check
                    try:
                        mongo_version = await self._run_command(['mongo', '--version'], timeout=5)
                        setup_result['installation_logs'].append("[OK] MongoDB installed")
                        setup_result['installation_logs'].append("[INFO] Start MongoDB with: mongod")
                    except Exception:
                        setup_result['installation_logs'].append("[WARN] MongoDB not found or not running")
                        setup_result['installation_logs'].append("[INFO] Install MongoDB: https://www.mongodb.com/try/download/community")
                        setup_result['errors'].append("MongoDB is required but not accessible")
            else:
                setup_result['database_setup'] = True  # Not needed
                
        except Exception as e:
            setup_result['errors'].append(f"MongoDB setup check failed: {str(e)}")
    
    async def _run_command(self, command: List[str], cwd: str = None, timeout: int = 30) -> str:
        """Run a command asynchronously and return output"""
        try:
            process = await asyncio.create_subprocess_exec(
                *command,
                cwd=cwd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
            
            if process.returncode == 0:
                return stdout.decode('utf-8')
            else:
                error_msg = stderr.decode('utf-8')
                raise Exception(f"Command failed: {' '.join(command)}\n{error_msg}")
                
        except asyncio.TimeoutError:
            raise Exception(f"Command timed out: {' '.join(command)}")
        except Exception as e:
            raise Exception(f"Command execution failed: {str(e)}")