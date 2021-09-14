/*!

=========================================================
* Argon Dashboard React - v1.2.1
=========================================================

* Product Page: https://www.creative-tim.com/product/argon-dashboard-react
* Copyright 2021 Creative Tim (https://www.creative-tim.com)
* Licensed under MIT (https://github.com/creativetimofficial/argon-dashboard-react/blob/master/LICENSE.md)

* Coded by Creative Tim

=========================================================

* The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

*/

import React from "react";
import {Redirect, Route, Switch, useLocation} from "react-router-dom";

// reactstrap components
import {Container} from "reactstrap";

import routes from "routes.js";
import Sidebar from "components/Sidebar/Sidebar.js";


const Admin = (props) => {

    const location = useLocation();
    const mainContent = React.useRef(null);

    React.useEffect(() => {
        document.documentElement.scrollTop = 0;
        document.scrollingElement.scrollTop = 0;
        mainContent.current.scrollTop = 0;
    }, [location]);

    const getRoutes = (routes) => {
        return routes.map((prop, key) => {
            if (prop.layout === "/admin") {
                return (
                    <Route
                        path={prop.layout + prop.path}
                        component={prop.component}
                        key={key}
                    />
                );
            } else {
                return null;
            }
        });
    };

    return (
        <>
            <Sidebar
                {...props}
                routes={routes}
                logo={{
                    innerLink: "/admin/index",
                    // imgSrc: require("../assets/images/brand/argon-react.png").default,
                    imgSrc: "/images/logo.png",
                    imgAlt: "...",
                }}
            />
            <div className="main-content" ref={mainContent}>
                <Switch>
                    {getRoutes(routes)}
                    <Redirect from="*" to="/admin/index"/>
                </Switch>
                <Container fluid/>
            </div>
        </>
    );
};

export default Admin;
