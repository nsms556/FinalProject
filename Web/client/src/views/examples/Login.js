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

import {useState} from "react";
import {Link} from "react-router-dom";

import axios from "axios";

// reactstrap components
import {
    Alert,
    Button,
    Card,
    CardBody,
    Col,
    Form,
    FormGroup,
    Input,
    InputGroup,
    InputGroupAddon,
    InputGroupText, Row
} from "reactstrap";

const Login = ({history}) => {

    const [Username, setUsername] = useState("");
    const [Password, setPassword] = useState("");

    const [CheckUser, setCheckUser] = useState(false);

    const onChangeUsername = (e) => {
        setUsername(e.currentTarget.value);
    }

    const onChangePassword = (e) => {
        setPassword(e.currentTarget.value);
    }

    const onSubmitHandler = (e) => {

        e.preventDefault();

        const user = {
            username: Username,
            password: Password
        };

        setCheckUser(false);

        axios.post('http://127.0.0.1:8000/users/login', user)
            .then(response => {
                if (response.data && response.data.success) {
                    history.push('/index');
                } else {
                    // alert('Fail to Login');
                    setCheckUser(true);
                }
            })
    }

    return (
        <>
            <Col lg="5" md="7">
                <Card className="bg-secondary shadow border-0">
                    <CardBody className="px-lg-5 py-lg-5">
                        <Form role="form" onSubmit={onSubmitHandler}>
                            {
                                CheckUser &&
                                <Alert color="danger">
                                    <span className="alert-inner--icon">
                                        <i className="ni ni-check-bold"/>
                                    </span>{" "}
                                    <span className="alert-inner--text">
                                        Couldn’t find your Account.
                                        <br/>{"　 "}
                                        or Wrong password. Try again.
                                    </span>
                                </Alert>
                            }
                            <FormGroup className="mb-3">
                                <InputGroup className="input-group-alternative">
                                    <InputGroupAddon addonType="prepend">
                                        <InputGroupText>
                                            <i className="ni ni-circle-08"/>
                                        </InputGroupText>
                                    </InputGroupAddon>
                                    <Input placeholder="Username" type="text" onChange={onChangeUsername}/>
                                </InputGroup>
                            </FormGroup>
                            <FormGroup>
                                <InputGroup className="input-group-alternative">
                                    <InputGroupAddon addonType="prepend">
                                        <InputGroupText>
                                            <i className="ni ni-lock-circle-open"/>
                                        </InputGroupText>
                                    </InputGroupAddon>
                                    <Input
                                        placeholder="Password"
                                        type="password"
                                        autoComplete="new-password"
                                        onChange={onChangePassword}
                                    />
                                </InputGroup>
                            </FormGroup>
                            <div className="text-center">
                                <Button className="my-4" color="primary" type="button" onClick={onSubmitHandler}>
                                    Sign in
                                </Button>
                            </div>
                        </Form>
                    </CardBody>
                </Card>
                <Row className="mt-3">
                    <Col xs="6"/>
                    <Col className="text-right" xs="6">
                        <Link className="text-light" to="/auth/register">
                            <small>Create new account</small>
                        </Link>
                    </Col>
                </Row>
            </Col>
        </>
    );
};

export default Login;
