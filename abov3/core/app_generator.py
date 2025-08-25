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
            
            # Step 4: Create deployment configurations
            deployment_config = await self._create_deployment_config(architecture)
            
            # Step 5: Generate documentation
            documentation = await self._generate_documentation(architecture, generation_result)
            
            return {
                'success': True,
                'analysis': analysis,
                'architecture': architecture,
                'generated_files': generation_result.get('files', []),
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
            
            # Determine website category
            if any(word in description_lower for word in ['shop', 'store', 'ecommerce', 'e-commerce', 'cart', 'buy', 'sell']):
                analysis['app_type'] = 'ecommerce'
                analysis['category'] = 'business'
                analysis['complexity'] = 'high'
                analysis['estimated_files'] = 25
            elif any(word in description_lower for word in ['restaurant', 'cafe', 'coffee', 'food', 'menu', 'dining']):
                analysis['app_type'] = 'restaurant'
                analysis['category'] = 'business'
                analysis['complexity'] = 'medium'
                analysis['estimated_files'] = 15
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
            if analysis['complexity'] == 'high' or 'shopping_cart' in analysis['features']:
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
        
        await self.code_generator.create_file(
            'frontend/package.json',
            json.dumps(frontend_package, indent=2),
            'Frontend package configuration'
        )
        
        # Generate main App.js
        app_js_content = self._generate_react_app_component(architecture)
        await self.code_generator.create_file('frontend/src/App.js', app_js_content, 'Main React App component')
        
        # Generate pages
        for page in architecture.get('pages', []):
            page_content = self._generate_react_page(page, architecture)
            page_name = f"{page.replace('-', '_').title().replace('_', '')}Page"
            await self.code_generator.create_file(
                f'frontend/src/pages/{page_name}.js',
                page_content,
                f'{page_name} component'
            )
            result['pages_created'] += 1
        
        # Generate components
        for component in architecture.get('components', []):
            component_content = self._generate_react_component(component, architecture)
            await self.code_generator.create_file(
                f'frontend/src/components/{component}.js',
                component_content,
                f'{component} React component'
            )
            result['components_created'] += 1
        
        # Generate backend
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
        await self.code_generator.create_file('index.html', index_content, 'Main HTML page')
        
        # Generate additional pages
        for page in architecture.get('pages', []):
            if page != 'home':
                page_content = self._generate_static_page_html(page, architecture)
                await self.code_generator.create_file(f'{page}.html', page_content, f'{page.title()} page')
                result['pages_created'] += 1
        
        # Generate CSS
        css_content = self._generate_static_css(architecture)
        await self.code_generator.create_file('styles/main.css', css_content, 'Main stylesheet')
        
        # Generate JavaScript
        js_content = self._generate_static_javascript(architecture)
        await self.code_generator.create_file('scripts/main.js', js_content, 'Main JavaScript file')
        
        result['files'].extend(['index.html', 'styles/main.css', 'scripts/main.js'])
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
   cd ../backend
   npm install
   ```

4. Set up environment variables:
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

5. Start MongoDB service

6. Initialize the database:
   ```bash
   npm run db:seed
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
npm run dev

# Terminal 2 - Frontend  
cd frontend
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